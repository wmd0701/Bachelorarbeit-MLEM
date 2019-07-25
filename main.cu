#include "algorithm"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "kernel.cuh"
#include "cusparse.h"
#include "csr4matrix.hpp"
#include "vector.hpp"
#include "time.h"
#include "sptrans.h"
#include "nccl.h"

void csr_format_for_cuda(const Csr4Matrix& matrix, float* csrVal, int* csrRowInd, int* csrColInd){   
    int index = 0;
    csrRowInd[index] = 0;
    int* tempIdx;
    tempIdx = (int*) malloc(sizeof(int) * matrix.rows());
    tempIdx[0] = 0;
    // !!! using openMP here will 100% lead to error in matrix
    // #pragma omp parallel for schedule (static)
    for (int row = 0; row < matrix.rows(); ++row) {
        csrRowInd[row + 1] = csrRowInd[row] + (int)matrix.elementsInRow(row);
        index += matrix.elementsInRow(row);
        tempIdx[row + 1] = index;
    }

    #pragma omp parallel for 
    for (int row = 0; row < matrix.rows(); ++row) {
            /*
             auto it = matrix.beginRow2(row);
             int count = 0;
             int localindex = index;
             #pragma omp parallel for reduction(+:count)
             for(int i=0; i< (matrix.endRow2(row) - it); i++){
                csrVal[localindex + i] = (it+i)->value();
                csrColInd[localindex + i] = (int)((it+i)->column());
                count++;
            }
            index += count;*/
            int idx=0;
            std::for_each(matrix.beginRow2(row), matrix.endRow2(row),[&](const RowElement<float>& e){ 
                csrVal[tempIdx[row]+idx] = e.value();
                csrColInd[tempIdx[row]+idx] = (int)e.column() ;
                idx++;
            }
               // index = index + 1; }
            );
    }
}

void calcColumnSums(const Csr4Matrix& matrix, Vector<float>& norm)
{
    assert(matrix.columns() == norm.size());

    std::fill(norm.ptr(), norm.ptr() + norm.size(), 0.0);
    matrix.mapRows(0, matrix.rows());

    #pragma omp declare reduction(vec_float_plus : std::vector<float> : \
        std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) \
        initializer(omp_priv = omp_orig)
    
    std::vector<float> res(norm.size(),0);
    #pragma omp parallel for ordered reduction(vec_float_plus:res)
    for (uint32_t row=0; row<matrix.rows(); ++row) {
        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                      [&](const RowElement<float>& e){ res[e.column()] += e.value(); });
    }
    #pragma omp parallel for 
    for(int i=0; i<norm.size(); i++){
        norm[i] = res[i];
    }

    // norm.writeToFile("norm-0.out");
}

/* a general version of halfMatrix: partition matrix into device_numbers parts, corresponding rows are saved in the array segments
   start row of segment i: segments[i]
    end  row of segment i: segments[i+1]
    number of rows in segment i: segments[i+1] - segments[i] (saved in segment_rows)
    number of nnzs in segment i: csr_Rows[segments[i+1]] - csr_Rows[segments[i]] (saved in segment_nnzs)
    offset when copying from host to device: csr_Rows[segments[i]] (saved in offsets)
*/
void partitionMatrix(int *csr_Rows, unsigned long nnzs, int rows, int device_numbers, int *segments, int *segment_rows, int *segment_nnzs, int *offsets){
    segments[0] = 0;
    segments[device_numbers] = rows;
    int i = 0;
    int nnzs_per_segment = (int)(nnzs / device_numbers);
    for(int segment = 1; segment < device_numbers; segment++){
        for(; i <= rows; i += 1)
            if(csr_Rows[i] >= nnzs_per_segment * segment)
                break;
        segments[segment] = i;
    }
    for(int segment = 0; segment < device_numbers; segment++){
        segment_rows[segment] = segments[segment+1] - segments[segment];
        segment_nnzs[segment] = csr_Rows[segments[segment+1]] - csr_Rows[segments[segment]];
        offsets[segment] = csr_Rows[segments[segment]];
    }
}


void mlem_nccl_none_trans(  int *csr_Rows, int *csr_Cols, float *csr_Vals, int *g, float *norm, float *f, float *result_f, 
                            int rows, int cols, unsigned long nnzs, 
                            int iterations, int device_numbers, int matrix_vector_mul, int secsize_fw){
    
    // partition matrix
    int *segments = (int*)malloc((device_numbers+1)*sizeof(int));
    int *segment_rows = (int*)malloc(device_numbers*sizeof(int));
    int *segment_nnzs = (int*)malloc(device_numbers*sizeof(int));
    int *offsets = (int*)malloc(device_numbers*sizeof(int));
    partitionMatrix(csr_Rows, nnzs, rows, device_numbers, segments, segment_rows, segment_nnzs, offsets);

    // NCCL elements
    ncclComm_t *comms = (ncclComm_t*)malloc(device_numbers * sizeof(ncclComm_t));;
    cudaStream_t *streams = (cudaStream_t*)malloc(device_numbers * sizeof(cudaStream_t));
    int *devices = (int*)malloc(device_numbers * sizeof(int));    

    // device variables
    int **cuda_Rows = (int**)malloc(device_numbers*sizeof(int*));
    int **cuda_Cols = (int**)malloc(device_numbers*sizeof(int*));
    int **cuda_g = (int**)malloc(device_numbers*sizeof(int*));
    float **cuda_Vals = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_norm = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_bwproj = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_temp = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_f = (float**)malloc(device_numbers*sizeof(float*));


    // initialization
    int blocksize = 1024;   // unique blocksize for all kernel calls
    int *gridsize_fwproj = (int*)malloc(device_numbers*sizeof(int));
    int *gridsize_correl = (int*)malloc(device_numbers*sizeof(int));
    int *gridsize_bwproj = (int*)malloc(device_numbers*sizeof(int));
    int *gridsize_update = (int*)malloc(device_numbers*sizeof(int));
    int *secsize_fwproj = (int*)malloc(device_numbers*sizeof(int));
    for(int i = 0; i < device_numbers; i++){
        cudaSetDevice(i);
        cudaStreamCreate(streams+i);
        devices[i] = i;

        cudaMalloc((void**)&cuda_Rows[i], sizeof(int)*(segment_rows[i] + 1));
        cudaMalloc((void**)&cuda_Cols[i], sizeof(int)*segment_nnzs[i]);
        cudaMalloc((void**)&cuda_Vals[i], sizeof(float)*segment_nnzs[i]);
        cudaMalloc((void**)&cuda_f[i], sizeof(float)*cols);
        cudaMalloc((void**)&cuda_bwproj[i], sizeof(float)*cols);
        cudaMalloc((void**)&cuda_norm[i], sizeof(float)*cols);
        cudaMalloc((void**)&cuda_temp[i], sizeof(float)*segment_rows[i]);
        cudaMalloc((void**)&cuda_g[i], sizeof(int)*segment_rows[i]);

        // copy matrix from host to devices
        for(int j = segments[i]; j <= segments[i+1]; j++)
            csr_Rows[j] -= offsets[i];
        cudaMemcpy(cuda_Rows[i], csr_Rows+segments[i], sizeof(int)*(segment_rows[i] + 1), cudaMemcpyHostToDevice);
        csr_Rows[segments[i+1]] += offsets[i];
        cudaMemcpy(cuda_Cols[i], csr_Cols+offsets[i], sizeof(int)*segment_nnzs[i], cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Vals[i], csr_Vals+offsets[i], sizeof(float)*segment_nnzs[i], cudaMemcpyHostToDevice);

        // copy other vectors from host to devices
        cudaMemcpy(cuda_g[i], g+segments[i], sizeof(int)*segment_rows[i], cudaMemcpyHostToDevice);
        cudaMemset(cuda_bwproj[i], 0, sizeof(float)*cols);
        cudaMemset(cuda_temp[i], 0, sizeof(float)*segment_rows[i]);
        cudaMemcpy(cuda_f[i], f, sizeof(float)*cols, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_norm[i], norm, sizeof(float)*cols, cudaMemcpyHostToDevice);

        // determine grid size for correlation and update
        gridsize_correl[i] = ceil((double)segment_rows[i] / blocksize);
        gridsize_update[i] = ceil((double)cols / blocksize);
        gridsize_bwproj[i] = ceil((double)segment_rows[i] / 32);
    }

    // NCCL initialization
    ncclCommInitAll(comms, device_numbers, devices);


    switch(matrix_vector_mul){
        case 0: { // NVIDIA merge-based
            // determine grid size and section size
            for(int i = 0; i < device_numbers; i++){
                int items_fwproj = segment_rows[i] + segment_nnzs[i];
                secsize_fwproj[i] = secsize_fw;
                gridsize_fwproj[i] = ceil((double)items_fwproj / (blocksize * secsize_fwproj[i]));
            }

            for(int iter = 0; iter < iterations; iter++){
                
                // forward projection
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcFwProj_merge_based <<< gridsize_fwproj[i], blocksize >>> (  cuda_Rows[i], cuda_Cols[i], cuda_Vals[i], cuda_f[i], 
                                                                                    cuda_temp[i], secsize_fwproj[i], segment_rows[i], segment_nnzs[i]);
                }

                // correlation
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcCorrel <<< gridsize_correl[i], blocksize >>> (cuda_g[i], cuda_temp[i], segment_rows[i]);
                }

                // backward projection
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcBwProj_none_trans <<< gridsize_bwproj[i], blocksize >>> (cuda_Rows[i], cuda_Cols[i], cuda_Vals[i], 
                                                                                cuda_temp[i], cuda_bwproj[i], segment_rows[i]);
                }

                // update, for mlem nccl calcUpdate should be used, followd by clearing bwproj using cudamemset
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcUpdate <<< gridsize_update[i], blocksize >>> (cuda_f[i], cuda_norm[i], cuda_bwproj[i], cols);
                }

                // sum up cuda_bwproj over devices and save in cuda_f
                ncclGroupStart();
                for (int i = 0; i < device_numbers; i++)
                    ncclAllReduce((const void*)cuda_bwproj[i], (void*)cuda_f[i], cols, ncclFloat, ncclSum, comms[i], streams[i]);
                ncclGroupEnd();
                for (int i = 0; i < device_numbers; ++i) {
                    cudaSetDevice(i);
                    cudaStreamSynchronize(streams[i]);
                }

                // clear cuda_bwproj
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    cudaMemset(cuda_bwproj[i], 0, sizeof(float)*cols);
                }

                // clear cuda_temp
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    cudaMemset(cuda_temp[i], 0, sizeof(float)*segment_rows[i]);
                }
            }
        } break;

        case 1: { // coalesced brutal warp
            for(int i = 0; i < device_numbers; i++)
                gridsize_fwproj[i] = ceil((double)segment_rows[i] / 32);
                
            
            for(int iter = 0; iter < iterations; iter++){
                
                // forward projection
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcFwProj_coalesced_brutal_warp <<< gridsize_fwproj[i], blocksize >>> (cuda_Rows[i], cuda_Cols[i], cuda_Vals[i], cuda_f[i], 
                                                                                            cuda_temp[i], segment_rows[i]);
                }

                // correlation
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcCorrel <<< gridsize_correl[i], blocksize >>> (cuda_g[i], cuda_temp[i], segment_rows[i]);
                }

                // backward projection
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcBwProj_none_trans <<< gridsize_bwproj[i], blocksize >>> (cuda_Rows[i], cuda_Cols[i], cuda_Vals[i], 
                                                                            cuda_temp[i], cuda_bwproj[i], segment_rows[i]);
                }
                
                // update, for mlem nccl calcUpdate should be used, followd by clearing bwproj using cudamemset
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcUpdate <<< gridsize_update[i], blocksize >>> (cuda_f[i], cuda_norm[i], cuda_bwproj[i], cols);
                }

                // sum up cuda_bwproj over devices and save in cuda_f
                ncclGroupStart();
                for (int i = 0; i < device_numbers; i++)
                    ncclAllReduce((const void*)cuda_bwproj[i], (void*)cuda_f[i], cols, ncclFloat, ncclSum, comms[i], streams[i]);
                ncclGroupEnd();
                for (int i = 0; i < device_numbers; ++i) {
                    cudaSetDevice(i);
                    cudaStreamSynchronize(streams[i]);
                }

                // clear cuda_bwproj
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    cudaMemset(cuda_bwproj[i], 0, sizeof(float)*cols);
                }

                // clear cuda_temp
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    cudaMemset(cuda_temp[i], 0, sizeof(float)*segment_rows[i]);
                }
            }
        } break;

        default: break;
    }


    for (int i = 0; i < device_numbers; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    // Result is copied to f from device 0, actually now all devices hold the same result
    
        cudaSetDevice(0);
        cudaMemcpy(result_f, cuda_f[0], sizeof(float)*cols, cudaMemcpyDeviceToHost);
        float sum = 0;
        for(unsigned long i = 0; i < cols; i += 1)
            sum += result_f[i];
        printf("\nSum f: %f\n", sum);
    

    // free all memory
    for(int i = 0; i < device_numbers; i++){
        cudaSetDevice(i);
        ncclCommDestroy(comms[i]);
        if(cuda_Rows[i]) cudaFree(cuda_Rows[i]);
        if(cuda_Cols[i]) cudaFree(cuda_Cols[i]);
        if(cuda_g[i]) cudaFree(cuda_g[i]);
        if(cuda_Vals[i]) cudaFree(cuda_Vals[i]);
        if(cuda_norm[i]) cudaFree(cuda_norm[i]);
        if(cuda_bwproj[i]) cudaFree(cuda_bwproj[i]);
        if(cuda_temp[i]) cudaFree(cuda_temp[i]);
        if(cuda_f[i]) cudaFree(cuda_f[i]);
    }
    if(segments) free(segments);
    if(segment_rows) free(segment_rows);
    if(segment_nnzs) free(segment_nnzs);
    if(offsets) free(offsets);
    if(comms) free(comms);
    if(streams) free(streams);
    if(devices) free(devices);
    if(cuda_Rows) free(cuda_Rows);
    if(cuda_Cols) free(cuda_Cols);
    if(cuda_g) free(cuda_g);
    if(cuda_Vals) free(cuda_Vals);
    if(cuda_norm) free(cuda_norm);
    if(cuda_bwproj) free(cuda_bwproj);
    if(cuda_temp) free(cuda_temp);
    if(cuda_f) free(cuda_f);
    if(gridsize_fwproj) free(gridsize_fwproj);
    if(gridsize_correl) free(gridsize_correl);
    if(gridsize_bwproj) free(gridsize_bwproj);
    if(gridsize_update) free(gridsize_update);
    if(secsize_fwproj) free(secsize_fwproj);
    
}


void mlem_nccl( int *csr_Rows, int *csr_Cols, float *csr_Vals,
                int *csr_Rows_Trans, int *csr_Cols_Trans, float *csr_Vals_Trans, 
                int *g, float *norm, float *f, float *result_f, int rows, int cols, unsigned long nnzs,
                int iterations, int device_numbers, int matrix_vector_mul, int secsize_fw, int secsize_bw){
    
    // partition matrix
    int *segments = (int*)malloc((device_numbers+1)*sizeof(int));
    int *segment_rows = (int*)malloc(device_numbers*sizeof(int));
    int *segment_nnzs = (int*)malloc(device_numbers*sizeof(int));
    int *offsets = (int*)malloc(device_numbers*sizeof(int));
    partitionMatrix(csr_Rows, nnzs, rows, device_numbers, segments, segment_rows, segment_nnzs, offsets);


    // partition transposed matrix
    int *segments_trans = (int*)malloc((device_numbers+1)*sizeof(int));
    int *segment_rows_trans = (int*)malloc(device_numbers*sizeof(int));
    int *segment_nnzs_trans = (int*)malloc(device_numbers*sizeof(int));
    int *offsets_trans = (int*)malloc(device_numbers*sizeof(int));
    partitionMatrix(csr_Rows_Trans, nnzs, cols, device_numbers, segments_trans, segment_rows_trans, segment_nnzs_trans, offsets_trans);

    
    // NCCL elements
    ncclComm_t *comms = (ncclComm_t*)malloc(device_numbers * sizeof(ncclComm_t));;
    cudaStream_t *streams = (cudaStream_t*)malloc(device_numbers * sizeof(cudaStream_t));
    int *devices = (int*)malloc(device_numbers * sizeof(int));    


    // device variables
    int **cuda_Rows = (int**)malloc(device_numbers*sizeof(int*));
    int **cuda_Cols = (int**)malloc(device_numbers*sizeof(int*));
    int **cuda_Rows_Trans = (int**)malloc(device_numbers*sizeof(int*));
    int **cuda_Cols_Trans = (int**)malloc(device_numbers*sizeof(int*)); 
    int **cuda_g = (int**)malloc(device_numbers*sizeof(int*));
    float **cuda_Vals = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_Vals_Trans = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_norm = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_bwproj = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_temp = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_f = (float**)malloc(device_numbers*sizeof(float*));
   

    // initialization
    int blocksize = 1024;   // unique blocksize for all kernel calls
    int *gridsize_fwproj = (int*)malloc(device_numbers*sizeof(int));
    int *gridsize_correl = (int*)malloc(device_numbers*sizeof(int));
    int *gridsize_bwproj = (int*)malloc(device_numbers*sizeof(int));
    int *gridsize_update = (int*)malloc(device_numbers*sizeof(int));
    int *secsize_fwproj = (int*)malloc(device_numbers*sizeof(int));
    int *secsize_bwproj = (int*)malloc(device_numbers*sizeof(int));
    for(int i = 0; i < device_numbers; i++){
        cudaSetDevice(i);
        cudaStreamCreate(streams+i);
        devices[i] = i;

        cudaMalloc((void**)&cuda_Rows[i], sizeof(int)*(segment_rows[i] + 1));
        cudaMalloc((void**)&cuda_Cols[i], sizeof(int)*segment_nnzs[i]);
        cudaMalloc((void**)&cuda_Vals[i], sizeof(float)*segment_nnzs[i]);
        cudaMalloc((void**)&cuda_Rows_Trans[i], sizeof(int)*(segment_rows_trans[i] + 1));
        cudaMalloc((void**)&cuda_Cols_Trans[i], sizeof(int)*segment_nnzs_trans[i]);
        cudaMalloc((void**)&cuda_Vals_Trans[i], sizeof(float)*segment_nnzs_trans[i]);
        cudaMalloc((void**)&cuda_f[i], sizeof(float)*cols);
        cudaMalloc((void**)&cuda_bwproj[i], sizeof(float)*cols);
        cudaMalloc((void**)&cuda_temp[i], sizeof(float)*rows);
        cudaMalloc((void**)&cuda_g[i], sizeof(int)*segment_rows[i]);
        cudaMalloc((void**)&cuda_norm[i], sizeof(float)*segment_rows_trans[i]);

        
        // copy matrix from host to devices
        for(int j = segments[i]; j <= segments[i+1]; j++)
            csr_Rows[j] -= offsets[i];
        cudaMemcpy(cuda_Rows[i], csr_Rows+segments[i], sizeof(int)*(segment_rows[i] + 1), cudaMemcpyHostToDevice);
        csr_Rows[segments[i+1]] += offsets[i];
        cudaMemcpy(cuda_Cols[i], csr_Cols+offsets[i], sizeof(int)*segment_nnzs[i], cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Vals[i], csr_Vals+offsets[i], sizeof(float)*segment_nnzs[i], cudaMemcpyHostToDevice);
        
        // copy transposed matrix from host to devices
        for(int j = segments_trans[i]; j <= segments_trans[i+1]; j++)
            csr_Rows_Trans[j] -= offsets_trans[i];
        cudaMemcpy(cuda_Rows_Trans[i], csr_Rows_Trans+segments_trans[i], sizeof(int)*(segment_rows_trans[i] + 1), cudaMemcpyHostToDevice);
        csr_Rows_Trans[segments_trans[i+1]] += offsets_trans[i];
        cudaMemcpy(cuda_Cols_Trans[i], csr_Cols_Trans+offsets_trans[i], sizeof(int)*segment_nnzs_trans[i], cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Vals_Trans[i], csr_Vals_Trans+offsets_trans[i], sizeof(float)*segment_nnzs_trans[i], cudaMemcpyHostToDevice);
        

        // copy other vectors from host to devices
        cudaMemcpy(cuda_g[i], g+segments[i], sizeof(int)*segment_rows[i], cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_norm[i], norm+segments_trans[i], sizeof(float)*segment_rows_trans[i], cudaMemcpyHostToDevice);
        cudaMemset(cuda_bwproj[i], 0, sizeof(float)*cols);
        cudaMemset(cuda_temp[i], 0, sizeof(float)*rows);
        cudaMemcpy(cuda_f[i], f, sizeof(float)*cols, cudaMemcpyHostToDevice);
        
        // determine grid size for correlation and update
        gridsize_correl[i] = ceil((double)segment_rows[i] / blocksize);
        gridsize_update[i] = ceil((double)segment_rows_trans[i] / blocksize);
    }

    // NCCL initialization
    ncclCommInitAll(comms, device_numbers, devices);


    switch(matrix_vector_mul){
        case 0: { // NVIDIA merge-based
            // determine grid size and section size
            for(int i = 0; i < device_numbers; i++){
                int items_fwproj = segment_rows[i] + segment_nnzs[i];
                int items_bwproj = segment_rows_trans[i] + segment_nnzs_trans[i];
                // determine section size for foward projection and backward projection
                secsize_fwproj[i] = secsize_fw; // ceil((double)items_fwproj / (blocksize * gridsize_fwproj[i]));
                secsize_bwproj[i] = secsize_bw; // ceil((double)items_bwproj / (blocksize * gridsize_bwproj[i]));
                // determine grid size for forward projection and backward projection
                gridsize_fwproj[i] = ceil((double)items_fwproj / (blocksize * secsize_fwproj[i])); // ceil(sqrt((double)items_fwproj / blocksize));
                gridsize_bwproj[i] = ceil((double)items_bwproj / (blocksize * secsize_bwproj[i])); // ceil(sqrt((double)items_bwproj / blocksize));      
            }
            

            for(int iter = 0; iter < iterations; iter++){
                
                // forward projection
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcFwProj_merge_based <<< gridsize_fwproj[i], blocksize >>> (  cuda_Rows[i], cuda_Cols[i], cuda_Vals[i], cuda_f[i], 
                                                                                    cuda_temp[i] + segments[i], secsize_fwproj[i], segment_rows[i], segment_nnzs[i]);
                }

                // correlation
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcCorrel <<< gridsize_correl[i], blocksize >>> (cuda_g[i], cuda_temp[i] + segments[i], segment_rows[i]);
                }

                // sum up cuda_temp over devices
                ncclGroupStart();
                for (int i = 0; i < device_numbers; i++)
                    ncclAllReduce((const void*)cuda_temp[i], (void*)cuda_temp[i], rows, ncclFloat, ncclSum, comms[i], streams[i]);
                ncclGroupEnd();
                for (int i = 0; i < device_numbers; ++i) {
                    cudaSetDevice(i);
                    cudaStreamSynchronize(streams[i]);
                }

                // backward projection
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcBwProj_merge_based <<< gridsize_bwproj[i], blocksize >>> (  cuda_Rows_Trans[i], cuda_Cols_Trans[i], cuda_Vals_Trans[i], cuda_temp[i], 
                                                                        cuda_bwproj[i] + segments_trans[i], secsize_bwproj[i], segment_rows_trans[i], segment_nnzs_trans[i]);
                }
            
                // update, for mlem nccl calcUpdate should be used, followd by clearing bwproj using cudamemset
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcUpdate <<< gridsize_update[i], blocksize >>> (cuda_f[i] + segments_trans[i], cuda_norm[i], cuda_bwproj[i] + segments_trans[i], segment_rows_trans[i]);
                }

                // sum up cuda_bwproj over devices and save in cuda_f
                ncclGroupStart();
                for (int i = 0; i < device_numbers; i++)
                    ncclAllReduce((const void*)cuda_bwproj[i], (void*)cuda_f[i], cols, ncclFloat, ncclSum, comms[i], streams[i]);
                ncclGroupEnd();
                for (int i = 0; i < device_numbers; ++i) {
                    cudaSetDevice(i);
                    cudaStreamSynchronize(streams[i]);
                }

                // clear cuda_bwproj
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    cudaMemset(cuda_bwproj[i], 0, sizeof(float)*cols);
                }

                // clear cuda_temp
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    cudaMemset(cuda_temp[i], 0, sizeof(float)*rows);
                }
            }
        } break;

        case 1: { // coalesced brutal warp
            for(int i = 0; i < device_numbers; i++){
                gridsize_fwproj[i] = ceil((double)segment_rows[i] / 32);
                gridsize_bwproj[i] = ceil((double)segment_rows_trans[i] / 32);
            }
            
            for(int iter = 0; iter < iterations; iter++){
                
                // forward projection
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcFwProj_coalesced_brutal_warp <<< gridsize_fwproj[i], blocksize >>> (cuda_Rows[i], cuda_Cols[i], cuda_Vals[i], cuda_f[i], 
                                                                                            cuda_temp[i] + segments[i], segment_rows[i]);
                }

                // correlation
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcCorrel <<< gridsize_correl[i], blocksize >>> (cuda_g[i], cuda_temp[i]+segments[i], segment_rows[i]);
                }

                // sum up cuda_temp over devices
                ncclGroupStart();
                for (int i = 0; i < device_numbers; i++)
                    ncclAllReduce((const void*)cuda_temp[i], (void*)cuda_temp[i], rows, ncclFloat, ncclSum, comms[i], streams[i]);
                ncclGroupEnd();
                for (int i = 0; i < device_numbers; ++i) {
                    cudaSetDevice(i);
                    cudaStreamSynchronize(streams[i]);
                }

                // backward projection
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcBwProj_coalesced_brutal_warp <<< gridsize_bwproj[i], blocksize >>> (cuda_Rows_Trans[i], cuda_Cols_Trans[i], cuda_Vals_Trans[i], 
                                                            cuda_temp[i], cuda_bwproj[i] + segments_trans[i], segment_rows_trans[i]);
                }
                
                // update, for mlem nccl calcUpdate should be used, followd by clearing bwproj using cudamemset
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcUpdate <<< gridsize_update[i], blocksize >>> (cuda_f[i] + segments_trans[i], cuda_norm[i], cuda_bwproj[i] + segments_trans[i], segment_rows_trans[i]);
                }

                // sum up cuda_bwproj over devices and save in cuda_f
                ncclGroupStart();
                for (int i = 0; i < device_numbers; i++)
                    ncclAllReduce((const void*)cuda_bwproj[i], (void*)cuda_f[i], cols, ncclFloat, ncclSum, comms[i], streams[i]);
                ncclGroupEnd();
                for (int i = 0; i < device_numbers; ++i) {
                    cudaSetDevice(i);
                    cudaStreamSynchronize(streams[i]);
                }

                // clear cuda_bwproj
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    cudaMemset(cuda_bwproj[i], 0, sizeof(float)*cols);
                }

                // clear cuda_temp
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    cudaMemset(cuda_temp[i], 0, sizeof(float)*rows);
                }
            }
        } break;

        default: break;
    }


    for (int i = 0; i < device_numbers; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    // Result is copied to f from device 0, actually now all devices hold the same result
    
        cudaSetDevice(0);
        cudaMemcpy(result_f, cuda_f[0], sizeof(float)*cols, cudaMemcpyDeviceToHost);
        float sum = 0;
        for(unsigned long i = 0; i < cols; i += 1)
            sum += result_f[i];
        printf("\nSum f: %f\n", sum);
    

    // free all memory
    for(int i = 0; i < device_numbers; i++){
        cudaSetDevice(i);
        ncclCommDestroy(comms[i]);
        if(cuda_Rows[i]) cudaFree(cuda_Rows[i]);
        if(cuda_Cols[i]) cudaFree(cuda_Cols[i]);
        if(cuda_Rows_Trans[i]) cudaFree(cuda_Rows_Trans[i]);
        if(cuda_Cols_Trans[i]) cudaFree(cuda_Cols_Trans[i]);
        if(cuda_g[i]) cudaFree(cuda_g[i]);
        if(cuda_Vals[i]) cudaFree(cuda_Vals[i]);
        if(cuda_Vals_Trans[i]) cudaFree(cuda_Vals_Trans[i]);
        if(cuda_norm[i]) cudaFree(cuda_norm[i]);
        if(cuda_bwproj[i]) cudaFree(cuda_bwproj[i]);
        if(cuda_temp[i]) cudaFree(cuda_temp[i]);
        if(cuda_f[i]) cudaFree(cuda_f[i]);
    }
    if(segments) free(segments);
    if(segment_rows) free(segment_rows);
    if(segment_nnzs) free(segment_nnzs);
    if(offsets) free(offsets);
    if(segments_trans) free(segments_trans);
    if(segment_rows_trans) free(segment_rows_trans);
    if(segment_nnzs_trans) free(segment_nnzs_trans);
    if(offsets_trans) free(offsets_trans);
    if(comms) free(comms);
    if(streams) free(streams);
    if(devices) free(devices);
    if(cuda_Rows) free(cuda_Rows);
    if(cuda_Cols) free(cuda_Cols);
    if(cuda_Rows_Trans) free(cuda_Rows_Trans);
    if(cuda_Cols_Trans) free(cuda_Cols_Trans);
    if(cuda_g) free(cuda_g);
    if(cuda_Vals) free(cuda_Vals);
    if(cuda_Vals_Trans) free(cuda_Vals_Trans);
    if(cuda_norm) free(cuda_norm);
    if(cuda_bwproj) free(cuda_bwproj);
    if(cuda_temp) free(cuda_temp);
    if(cuda_f) free(cuda_f);
    if(gridsize_fwproj) free(gridsize_fwproj);
    if(gridsize_correl) free(gridsize_correl);
    if(gridsize_bwproj) free(gridsize_bwproj);
    if(gridsize_update) free(gridsize_update);
    if(secsize_fwproj) free(secsize_fwproj);
    if(secsize_bwproj) free(secsize_bwproj);
    
}


/*
    argv[1]: path for matrix
    argv[2]: path for image
    argv[3]: iteration times
    argv[4]: number of GPUs to be used
    argv[5]: section size for forward projection in NVIDIA merge-based
    argv[6]: section size for backward projection in NVIDIA merge-based
    argv[7]: whether to use transposed matrix              0: use transposed matrix    1: not use transposed matrix
    argv[8]: which matrix-vector multiplication to use     0: NVIDIA merge-based       1: coalesced brutal warp

    run examples:
    ./test /scratch/pet/madpet2.p016.csr4.small /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino.small 500 2 4 4 0 0
    ./test /scratch/pet/madpet2.p016.csr4.small /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino.small 500 2 3 9 1 0
    ./test /scratch/pet/madpet2.p016.csr4.small /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino.small 500 2 3 9 0 1
    ./test /scratch/pet/madpet2.p016.csr4.small /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino.small 500 2 5 5 1 1
    ./test /scratch/pet/madpet2.p016.csr4 /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino 500 1 5 5 1 1
*/
int main(int argc, char **argv){
    if(argc != 9){
        printf("Too less or too many parameters for main function! Program exits!\n");
        return 0;
    }

    std::string matrixPath(argv[1]);
    std::string imagePath(argv[2]);
    int iterations          = strtol(argv[3], NULL, 10);
    int device_numbers      = strtol(argv[4], NULL, 10);
    int secsize_fw          = strtol(argv[5], NULL, 10);
    int secsize_bw          = strtol(argv[6], NULL, 10);
    int using_trans         = strtol(argv[7], NULL, 10);
    int matrix_vector_mul   = strtol(argv[8], NULL, 10);
    /*
    if(device_numbers < 2){
        printf("Less than 2 GPUs are used! Program exits!\n");
        return 0;
    }
    */
    int device_numbers_available = 0;
    cudaGetDeviceCount(&device_numbers_available);
    if(device_numbers_available < device_numbers){
        printf("Number of available GPUs less than ordered! Program exits!\n");
        return 0;
    }

    // host variables
    int *csr_Rows, *csr_Cols, *csr_Rows_Trans, *csr_Cols_Trans, *g, sum_g = 0, rows, cols;
    unsigned long nnzs;
    float *csr_Vals, *csr_Vals_Trans, *f, *result_f, *norm, sum_norm = 0.0f;


    // read matrix
    Csr4Matrix matrix(matrixPath);
    rows = matrix.rows();
    cols = matrix.columns();
    nnzs = matrix.elements();
    matrix.mapRows(0, rows);    
    csr_Rows = (int*)malloc(sizeof(int) * (rows + 1));
    csr_Cols = (int*)malloc(sizeof(int) * nnzs);
    csr_Vals = (float*)malloc(sizeof(float) * nnzs);
    csr_format_for_cuda(matrix, csr_Vals, csr_Rows, csr_Cols);
    Vector<float> norm_helper(cols, 0.0);
    calcColumnSums(matrix, norm_helper);
    norm = norm_helper.ptr();
    for(int i = 0; i < cols; i++)
        sum_norm += norm[i];
    

    // read image
    Vector<int> image(imagePath);
    g = image.ptr();
    for(int i = 0; i < rows; i++)
        sum_g += g[i];
    
    // calculate initial value
    float init = sum_g / sum_norm;
    f = (float*)malloc(sizeof(float)*cols);
    result_f = (float*)malloc(sizeof(float)*cols);
    for(int i = 0; i < cols; i++)
        f[i] = init;


    // transpose matrix using CPU
    if(using_trans == 0){
        csr_Rows_Trans = (int*) calloc (cols+1,sizeof(int));
        csr_Cols_Trans = (int*) calloc (nnzs,sizeof(int));
        csr_Vals_Trans = (float*) calloc (nnzs,sizeof(float));
        sptrans_scanTrans<int, float>(rows, cols, nnzs, csr_Rows, csr_Cols, csr_Vals, csr_Cols_Trans, csr_Rows_Trans, csr_Vals_Trans);
    }

        
    // run mlem function
    if(using_trans == 0)
        mlem_nccl(csr_Rows, csr_Cols, csr_Vals, csr_Rows_Trans, csr_Cols_Trans, csr_Vals_Trans, g, norm, f, result_f, rows, cols, nnzs, 
                iterations, device_numbers, matrix_vector_mul, secsize_fw, secsize_bw); 
    else
        mlem_nccl_none_trans(csr_Rows, csr_Cols, csr_Vals, g, norm, f, result_f, rows, cols, nnzs,
                iterations, device_numbers, matrix_vector_mul, secsize_fw);


    // clear storage
    if (csr_Rows) free(csr_Rows);
    if (csr_Cols) free(csr_Cols);
    if (csr_Vals) free(csr_Vals);
    // if (g) free(g);
    // if (norm) free(norm);
    if (f) free(f);
    if(result_f) free(result_f);
    if(using_trans == 0){
        if (csr_Rows_Trans) free(csr_Rows_Trans);
        if (csr_Cols_Trans) free(csr_Cols_Trans);
        if (csr_Vals_Trans) free(csr_Vals_Trans);
    }

    return 0;
}
