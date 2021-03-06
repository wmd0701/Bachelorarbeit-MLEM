#include "algorithm"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "kernel_large.cuh"
#include "cusparse.h"
#include "csr4matrix.hpp"
#include "vector.hpp"
#include "time.h"
#include "sptrans.h"
#include "nccl.h"

void csr_format_for_cuda(const Csr4Matrix& matrix, float* csrVal, unsigned long* csrRowInd, unsigned int* csrColInd){   
    unsigned int index = 0;
    csrRowInd[index] = 0;
    unsigned int* tempIdx;
    tempIdx = (unsigned int*) malloc(sizeof(unsigned int) * matrix.rows());
    tempIdx[0] = 0;
    // !!! using openMP here will 100% lead to error in matrix
    // #pragma omp parallel for schedule (static)
    for (unsigned int row = 0; row < matrix.rows(); ++row) {
        csrRowInd[row + 1] = csrRowInd[row] + matrix.elementsInRow(row);
        index += matrix.elementsInRow(row);
        tempIdx[row + 1] = index;
    }

    #pragma omp parallel for 
    for (unsigned int row = 0; row < matrix.rows(); ++row) {
            /*
             auto it = matrix.beginRow2(row);
             unsigned int count = 0;
             unsigned int localindex = index;
             #pragma omp parallel for reduction(+:count)
             for(unsigned int i=0; i< (matrix.endRow2(row) - it); i++){
                csrVal[localindex + i] = (it+i)->value();
                csrColInd[localindex + i] = (unsigned int)((it+i)->column());
                count++;
            }
            index += count;*/
            unsigned int idx=0;
            std::for_each(matrix.beginRow2(row), matrix.endRow2(row),[&](const RowElement<float>& e){ 
                csrVal[tempIdx[row]+idx] = e.value();
                csrColInd[tempIdx[row]+idx] = (unsigned int)e.column() ;
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
    for(unsigned int i=0; i<norm.size(); i++){
        norm[i] = res[i];
    }

    // norm.writeToFile("norm-0.out");
}

/* a general version of halfMatrix: partition matrix unsigned into device_numbers parts, corresponding rows are saved in the array segments
   start row of segment i: segments[i]
    end  row of segment i: segments[i+1]
    number of rows in segment i: segments[i+1] - segments[i] (saved in segment_rows)
    number of nnzs in segment i: csr_Rows[segments[i+1]] - csr_Rows[segments[i]] (saved in segment_nnzs)
    offset when copying from host to device: csr_Rows[segments[i]] (saved in offsets)
*/
void partitionMatrix(unsigned long *csr_Rows, unsigned long nnzs, unsigned int rows, unsigned int device_numbers, unsigned int *segments, unsigned int *segment_rows, unsigned int *segment_nnzs, unsigned long *offsets){
    segments[0] = 0;
    segments[device_numbers] = rows;
    unsigned int i = 0;
    double nnzs_per_segment = ((double)nnzs / (double)device_numbers);
    for(unsigned int segment = 0; segment < device_numbers; segment++){
        int sum = 0;
        for(; i <= rows; i += 1){
            if(csr_Rows[i] > nnzs_per_segment * segment){
                printf("DEBUG: csr_Rows %u > nnzs_per_segments %u * segments %d\n",  csr_Rows[i], nnzs_per_segment, segment);
                break;
            }
        }
        segments[segment] = i;
    }
    for(unsigned int segment = 0; segment < device_numbers; segment++){
        segment_rows[segment] = segments[segment+1] - segments[segment];
        segment_nnzs[segment] = (unsigned int)(csr_Rows[segments[segment+1]] - csr_Rows[segments[segment]]);
        offsets[segment] = csr_Rows[segments[segment]];
        printf("Segment %u with rows: %u nnzs %u offset %u\n", segment, segment_rows[segment], segment_nnzs[segment], offsets[segment]);
    }
}


void mlem_nccl_none_trans(  unsigned long *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, int *g, float *norm, float *f, float *result_f, 
                            unsigned int rows, unsigned int cols, unsigned long nnzs, 
                            unsigned int iterations, unsigned int device_numbers){
    
    // partition matrix
    unsigned int *segments = (unsigned int*)malloc((device_numbers+1)*sizeof(unsigned int));
    unsigned int *segment_rows = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
    unsigned int *segment_nnzs = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
    unsigned long *offsets = (unsigned long*)malloc(device_numbers*sizeof(unsigned long));
    partitionMatrix(csr_Rows, nnzs, rows, device_numbers, segments, segment_rows, segment_nnzs, offsets);

    // NCCL elements
    ncclComm_t *comms = (ncclComm_t*)malloc(device_numbers * sizeof(ncclComm_t));;
    cudaStream_t *streams = (cudaStream_t*)malloc(device_numbers * sizeof(cudaStream_t));
    int *devices = (int*)malloc(device_numbers * sizeof(int));    

    // device variables
    unsigned int **cuda_Rows = (unsigned int**)malloc(device_numbers*sizeof(unsigned int*));
    unsigned int **cuda_Cols = (unsigned int**)malloc(device_numbers*sizeof(unsigned int*));
    int **cuda_g = (int**)malloc(device_numbers*sizeof(int*));
    float **cuda_Vals = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_norm = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_bwproj = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_temp = (float**)malloc(device_numbers*sizeof(float*));
    float **cuda_f = (float**)malloc(device_numbers*sizeof(float*));


    // initialization
    unsigned int blocksize = 1024;   // unique blocksize for all kernel calls
    unsigned int *gridsize_fwproj = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
    unsigned int *gridsize_correl = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
    unsigned int *gridsize_bwproj = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
    unsigned int *gridsize_update = (unsigned int*)malloc(device_numbers*sizeof(unsigned int));
    for(unsigned int i = 0; i < device_numbers; i++){
        cudaSetDevice(i);
        cudaStreamCreate(streams+i);
        devices[i] = i;

        cudaMalloc((void**)&cuda_Rows[i], sizeof(unsigned int)*(segment_rows[i] + 1));
        cudaMalloc((void**)&cuda_Cols[i], sizeof(unsigned int)*segment_nnzs[i]);
        cudaMalloc((void**)&cuda_Vals[i], sizeof(float)*segment_nnzs[i]);
        cudaMalloc((void**)&cuda_f[i], sizeof(float)*cols);
        cudaMalloc((void**)&cuda_bwproj[i], sizeof(float)*cols);
        cudaMalloc((void**)&cuda_norm[i], sizeof(float)*cols);
        cudaMalloc((void**)&cuda_temp[i], sizeof(float)*segment_rows[i]);
        cudaMalloc((void**)&cuda_g[i], sizeof(int)*segment_rows[i]);

        // copy matrix from host to devices
        for(unsigned int j = segments[i]; j <= segments[i+1]; j++)
            csr_Rows[j] -= offsets[i];
        unsigned int *csr_Rows_help = (unsigned int*)malloc((segment_rows[i] + 1)*sizeof(unsigned int));
        for(unsigned int j = 0; j < segment_rows[i] + 1; j++)
            csr_Rows_help[j] = (unsigned int)csr_Rows[segments[i]+j];
        // cudaMemcpy(cuda_Rows[i], csr_Rows+segments[i], sizeof(unsigned int)*(segment_rows[i] + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Rows[i], csr_Rows_help, sizeof(unsigned int)*(segment_rows[i] + 1), cudaMemcpyHostToDevice);
        csr_Rows[segments[i+1]] += offsets[i];

        // test
        printf("Number of rows on GPU %d is: %u\n", i, segment_rows[i]);
        printf("Number of nnzs on GPU %d is: %u\n", i, segment_nnzs[i]);
        printf("First element in csr_Rows on GPU %d is: %d\n", i, csr_Rows_help[0]);
        printf("Last element in csr_Rows on GPU %d is: %d\n\n", i, csr_Rows_help[segment_rows[i] + 1]);


        free(csr_Rows_help);
        cudaMemcpy(cuda_Cols[i], csr_Cols+offsets[i], sizeof(unsigned int)*segment_nnzs[i], cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Vals[i], csr_Vals+offsets[i], sizeof(float)*segment_nnzs[i], cudaMemcpyHostToDevice);

        // copy other vectors from host to devices
        cudaMemcpy(cuda_g[i], g+segments[i], sizeof(int)*segment_rows[i], cudaMemcpyHostToDevice);
        cudaMemset(cuda_bwproj[i], 0, sizeof(float)*cols);
        cudaMemset(cuda_temp[i], 0, sizeof(float)*segment_rows[i]);
        cudaMemcpy(cuda_f[i], f, sizeof(float)*cols, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_norm[i], norm, sizeof(float)*cols, cudaMemcpyHostToDevice);

        // determine grid size
        gridsize_correl[i] = ceil((double)segment_rows[i] / blocksize);
        gridsize_update[i] = ceil((double)cols / blocksize);
        gridsize_bwproj[i] = ceil((double)segment_rows[i] / 32);
        gridsize_fwproj[i] = ceil((double)segment_rows[i] / 32);
    }

    // NCCL initialization
    ncclCommInitAll(comms, device_numbers, devices);

    
    // iterations
    for(unsigned int iter = 0; iter < iterations; iter++){
        // forward projection
        for(unsigned int i = 0; i < device_numbers; i++){
            cudaSetDevice(i);
            calcFwProj_coalesced_brutal_warp <<< gridsize_fwproj[i], blocksize >>> (cuda_Rows[i], cuda_Cols[i], cuda_Vals[i], cuda_f[i], 
                                                                                    cuda_temp[i], segment_rows[i]);
        }

        // correlation
        for(unsigned int i = 0; i < device_numbers; i++){
            cudaSetDevice(i);
            calcCorrel <<< gridsize_correl[i], blocksize >>> (cuda_g[i], cuda_temp[i], segment_rows[i]);
        }

        // backward projection
        for(unsigned int i = 0; i < device_numbers; i++){
            cudaSetDevice(i);
            calcBwProj_none_trans <<< gridsize_bwproj[i], blocksize >>> (cuda_Rows[i], cuda_Cols[i], cuda_Vals[i], 
                                                                    cuda_temp[i], cuda_bwproj[i], segment_rows[i]);
        }
        
        // update, for mlem nccl calcUpdate should be used, followd by clearing bwproj using cudamemset
        for(unsigned int i = 0; i < device_numbers; i++){
            cudaSetDevice(i);
            calcUpdate <<< gridsize_update[i], blocksize >>> (cuda_f[i], cuda_norm[i], cuda_bwproj[i], cols);
        }

        // sum up cuda_bwproj over devices and save in cuda_f
        ncclGroupStart();
        for (unsigned int i = 0; i < device_numbers; i++)
            ncclAllReduce((const void*)cuda_bwproj[i], (void*)cuda_f[i], cols, ncclFloat, ncclSum, comms[i], streams[i]);
        ncclGroupEnd();
        for (unsigned int i = 0; i < device_numbers; ++i) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
        }

        // clear cuda_bwproj
        for(unsigned int i = 0; i < device_numbers; i++){
            cudaSetDevice(i);
            cudaMemset(cuda_bwproj[i], 0, sizeof(float)*cols);
        }

        // clear cuda_temp
        for(unsigned int i = 0; i < device_numbers; i++){
            cudaSetDevice(i);
            cudaMemset(cuda_temp[i], 0, sizeof(float)*segment_rows[i]);
        }
    } 


    // synchronize GPUs
    for (unsigned int i = 0; i < device_numbers; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    // Result is copied to f from device 0, actually now all devices hold the same result
    cudaSetDevice(0);
    cudaMemcpy(result_f, cuda_f[0], sizeof(float)*cols, cudaMemcpyDeviceToHost);
    float sum = 0;
    for(unsigned long i = 0; i < cols; i += 1)
        sum += result_f[i];
    printf("\nSum f: %f\n\n", sum);
    

    // free all memory
    for(unsigned int i = 0; i < device_numbers; i++){
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
}


/*
    argv[1]: path for matrix
    argv[2]: path for image
    argv[3]: iteration times
    argv[4]: number of GPUs to be used
    
    coalesced brutal warp will be used by default 

    run examples:
    ./test /scratch/pet/madpet2.p016.csr4.small /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino.small 500 1
    ./test /scratch/pet/madpet2.p016.csr4.small /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino.small 500 2
    ./test /scratch/pet/madpet2.p016.csr4 /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino 50 1
    ./test /scratch/pet/madpet2.p016.csr4 /scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino 500 1
*/
int main(int argc, char **argv){
    if(argc != 5){
        printf("Too less or too many parameters for main function! Program exits!\n");
        return 0;
    }

    std::string matrixPath(argv[1]);
    std::string imagePath(argv[2]);
    unsigned int iterations          = strtol(argv[3], NULL, 10);
    unsigned int device_numbers      = strtol(argv[4], NULL, 10);
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
    unsigned int *csr_Cols, rows, cols;
    int *g, sum_g = 0;
    unsigned long *csr_Rows, nnzs;
    float *csr_Vals, *f, *result_f, *norm, sum_norm = 0.0f;


    // read matrix
    Csr4Matrix matrix(matrixPath);
    rows = matrix.rows();
    cols = matrix.columns();
    nnzs = matrix.elements();
    matrix.mapRows(0, rows);    
    csr_Rows = (unsigned long*)malloc(sizeof(unsigned long) * (rows + 1));
    csr_Cols = (unsigned int*)malloc(sizeof(unsigned int) * nnzs);
    csr_Vals = (float*)malloc(sizeof(float) * nnzs);
    csr_format_for_cuda(matrix, csr_Vals, csr_Rows, csr_Cols);
    Vector<float> norm_helper(cols, 0.0);
    calcColumnSums(matrix, norm_helper);
    norm = norm_helper.ptr();
    for(unsigned int i = 0; i < cols; i++)
        sum_norm += norm[i];
    

    // read image
    Vector<int> image(imagePath);
    g = image.ptr();
    for(unsigned int i = 0; i < rows; i++)
        sum_g += g[i];
    
    // calculate initial value
    float init = sum_g / sum_norm;
    f = (float*)malloc(sizeof(float)*cols);
    result_f = (float*)malloc(sizeof(float)*cols);
    for(unsigned int i = 0; i < cols; i++)
        f[i] = init;


    // run mlem function
    mlem_nccl_none_trans(csr_Rows, csr_Cols, csr_Vals, g, norm, f, result_f, rows, cols, nnzs,iterations, device_numbers);


    // clear storage
    if (csr_Rows) free(csr_Rows);
    if (csr_Cols) free(csr_Cols);
    if (csr_Vals) free(csr_Vals);
    // if (g) free(g);
    // if (norm) free(norm);
    if (f) free(f);
    if(result_f) free(result_f);

    return 0;
}
