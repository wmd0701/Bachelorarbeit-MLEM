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
    printf("Begin: CSR_FORMAT_FOR_CUDA\n");
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
    printf("End  : CSR_FORMAT_FOR_CUDA\n");
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

void transposeCSR(int *cuda_Rows, int *cuda_Cols, float *cuda_Vals, int *cuda_Rows_Trans, int *cuda_Cols_Trans, float *cuda_Vals_Trans,
                    int rows, int cols, int nnzs){
    cusparseStatus_t status;
	cusparseHandle_t handle = 0;
	status = cusparseCreate(&handle);
	if (status != CUSPARSE_STATUS_SUCCESS){
        cudaError_t cuda_err = cudaGetLastError();
        printf("    Fail : CSR to CSC, cusparese initialization failed , ERROR %d, %s\n", status, cudaGetErrorString(cuda_err));
    }
    status = cusparseScsr2csc(handle, rows, cols, nnzs, cuda_Vals, cuda_Rows, cuda_Cols, cuda_Vals_Trans, cuda_Cols_Trans, cuda_Rows_Trans, 
                                CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    if (status != CUSPARSE_STATUS_SUCCESS)
        printf("    Fail : CSR to CSC, cusparse transpose failed\n");

    status = cusparseDestroy(handle);
    handle = 0;
	if (status != CUSPARSE_STATUS_SUCCESS)
        printf("    Fail : CSR to CSC, cusparse destroy failed\n");
    
    // cusparse functions are asynchronous
    cudaDeviceSynchronize();
}


// return row index for in which row the nnzs are distributed into two pars equally
int halfMatrix(int *csr_Rows, int nnzs, int rows){
    int i = 0;
    int halfnnzs = nnzs / 2;
    for(; i <= rows; i++)
        if(csr_Rows[i] >= halfnnzs)
            break;
    return i;
}

// return row index for in which row the nnzs are distributed into two pars equally
int fiveSixth(int *csr_Rows, int nnzs, int rows){
    int i = 0;
    double halfnnzs = (double)nnzs * 5.0 / 6.0;
    for(; i <= rows; i++)
        if(csr_Rows[i] >= halfnnzs)
            break;
    
    return i;
}

/* a general version of halfMatrix: partition matrix into device_numbers parts, corresponding rows are saved in the array segments
   start row of segment i: segments[i]
    end  row of segment i: segments[i+1]
    number of rows in segment i: segments[i+1] - segments[i] (saved in segment_rows)
    number of nnzs in segment i: csr_Rows[segments[i+1]] - csr_Rows[segments[i]] (saved in segment_nnzs)
    offset when copying from host to device: csr_Rows[segments[i]] (saved in offsets)
*/
void partitionMatrix(int *csr_Rows, int nnzs, int rows, int device_numbers, int *segments, int *segment_rows, int *segment_nnzs, int *offsets){
    segments[0] = 0;
    segments[device_numbers] = rows;
    int i = 0;
    int nnzs_per_segment = nnzs / device_numbers;
    for(int segment = 1; segment < device_numbers; segment++){
        for(; i <= rows; i++)
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


void mlem_nccl_none_trans( int *csr_Rows, int *csr_Cols, float *csr_Vals, int *g, float *norm, float *f, float *result_f, int rows, int cols, int nnzs){
    int scan_result, device_numbers = 2;

    cudaGetDeviceCount(&device_numbers);
    if(device_numbers < 2){
        printf("\nWarning! Number of capable GPUs less than 2! Program exits!\n\n");
        return;
    }
    else{
        printf("\nNumber of usable GPUs: %d\n", device_numbers);
        printf("How many GPUs to use? Please give a number bigger than or equal to 2: ");
        scan_result = scanf("%d", &device_numbers);
        printf("Running NCCL MLEM with %d CUDA devices\n\n", device_numbers);
    }

    clock_t start = clock();
    printf("Begin: Initialization\n");
    clock_t initStart = clock();

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

    clock_t initEnd = clock();
    printf("End  : Initialization\n");
    double initTime = ((double) (initEnd - initStart)) / CLOCKS_PER_SEC;
    printf("Elapsed time for initialization: %f\n\n", initTime);


    int iterations = 50;
    printf("\nIteration times: ");
    scan_result = scanf("%d", &iterations);

    int matrix_vector_mul = 0;
    printf("\nUse which kind of matrix-vector multiplication? (0: CSRMV   1: coalesced brutal warp): ");
    scan_result = scanf("%d", &matrix_vector_mul);
    printf("\n");

    clock_t iterStart;
    switch(matrix_vector_mul){
        case 0: { // CSRMV
            int secsize_fw = 8;
            printf("    default section size forward projection is %d\n", secsize_fw);
            printf("    please give new section size forward projection: ");
            scan_result = scanf("%d", &secsize_fw);
            printf("    remember! You are using no transposed matrix, thus no section size backward projection needed\n\n");

            // determine grid size and section size
            for(int i = 0; i < device_numbers; i++){
                int items_fwproj = segment_rows[i] + segment_nnzs[i];
                secsize_fwproj[i] = secsize_fw;
                gridsize_fwproj[i] = ceil((double)items_fwproj / (blocksize * secsize_fwproj[i]));
            }

            printf("Begin: Iterations %d\n", iterations);
            iterStart = clock();

            for(int iter = 0; iter < iterations; iter++){
                
                // forward projection
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcFwProj <<< gridsize_fwproj[i], blocksize >>> (  cuda_Rows[i], cuda_Cols[i], cuda_Vals[i], cuda_f[i], 
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
            } break;
        }

        case 1: { // coalesced brutal warp
            for(int i = 0; i < device_numbers; i++)
                gridsize_fwproj[i] = ceil((double)segment_rows[i] / 32);
                
            printf("Begin: Iterations %d\n", iterations);
            iterStart = clock();

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
    clock_t iterEnd = clock();
    printf("End  : Iterations %d\n", iterations);
    double itertime = ((double) (iterEnd - iterStart)) / CLOCKS_PER_SEC;
    printf("Elapsed time for iterations: %f\n\n", itertime);


    // Result is copied to f from device 0, actually now all devices hold the same result
    cudaSetDevice(0);
    cudaMemcpy(result_f, cuda_f[0], sizeof(float)*cols, cudaMemcpyDeviceToHost);
    float sum = 0;
    for(int i = 0; i < cols; i++)
        sum += result_f[i];
    printf("\nSum f: %f\n\n", sum);

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
    

    clock_t end = clock();
    double totaltime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time totally       : %f\n\n", totaltime);
}


void mlem_nccl( int *csr_Rows, int *csr_Cols, float *csr_Vals,
                int *csr_Rows_Trans, int *csr_Cols_Trans, float *csr_Vals_Trans, 
                int *g, float *norm, float *f, float *result_f, int rows, int cols, int nnzs){
    
    int scan_result, device_numbers = 2;

    cudaGetDeviceCount(&device_numbers);
    if(device_numbers < 2){
        printf("\nWarning! Number of capable GPUs less than 2! Program exits!\n\n");
        return;
    }
    else{
        printf("\nNumber of usable GPUs: %d\n", device_numbers);
        printf("How many GPUs to use? Please give a number bigger than or equal to 2: ");
        scan_result = scanf("%d", &device_numbers);
        printf("Running NCCL MLEM with %d CUDA devices\n\n", device_numbers);
    }

    clock_t start = clock();
    printf("Begin: Initialization\n");
    clock_t initStart = clock();

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

    clock_t initEnd = clock();
    printf("End  : Initialization\n");
    double initTime = ((double) (initEnd - initStart)) / CLOCKS_PER_SEC;
    printf("Elapsed time for initialization: %f\n\n", initTime);


    int iterations = 50;
    printf("\nIteration times: ");
    scan_result = scanf("%d", &iterations);

    int matrix_vector_mul = 0;
    printf("\nUse which kind of matrix-vector multiplication? (0: CSRMV   1: coalesced brutal warp): ");
    scan_result = scanf("%d", &matrix_vector_mul);

    clock_t iterStart;
    switch(matrix_vector_mul){
        case 0: { // CSRMV
            int secsize_fw = 8, secsize_bw = 8;
            printf("    default section size forward projection is %d\n", secsize_fw);
            printf("    please give new section size forward projection: ");
            scan_result = scanf("%d", &secsize_fw);
            printf("    default section size backward projection is %d\n", secsize_bw);
            printf("    please give new section size backward projection (useless when using no trans matrix): ");
            scan_result = scanf("%d", &secsize_bw);
            printf("\n");

            // determine grid size and section size
            for(int i = 0; i < device_numbers; i++){
                int items_fwproj = segment_rows[i] + segment_nnzs[i];
                int items_bwproj = segment_rows_trans[i] + segment_nnzs_trans[i];
                // determine section size for foward projection and backward projection
                secsize_fwproj[i] = secsize_fw; // ceil((double)items_fwproj / (blocksize * gridsize_fwproj[i]));
                secsize_bwproj[i] = secsize_fw; // ceil((double)items_bwproj / (blocksize * gridsize_bwproj[i]));
                // determine grid size for forward projection and backward projection
                gridsize_fwproj[i] = ceil((double)items_fwproj / (blocksize * secsize_fwproj[i])); // ceil(sqrt((double)items_fwproj / blocksize));
                gridsize_bwproj[i] = ceil((double)items_bwproj / (blocksize * secsize_bwproj[i])); // ceil(sqrt((double)items_bwproj / blocksize));      
            }

            printf("Begin: Iterations %d\n", iterations);
            iterStart = clock();
            
            for(int iter = 0; iter < iterations; iter++){
                
                // forward projection
                for(int i = 0; i < device_numbers; i++){
                    cudaSetDevice(i);
                    calcFwProj <<< gridsize_fwproj[i], blocksize >>> (  cuda_Rows[i], cuda_Cols[i], cuda_Vals[i], cuda_f[i], 
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
                    calcBwProj <<< gridsize_bwproj[i], blocksize >>> (  cuda_Rows_Trans[i], cuda_Cols_Trans[i], cuda_Vals_Trans[i], cuda_temp[i], 
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
            } break;
        }

        case 1: { // coalesced brutal warp
            for(int i = 0; i < device_numbers; i++){
                gridsize_fwproj[i] = ceil((double)segment_rows[i] / 32);
                gridsize_bwproj[i] = ceil((double)segment_rows_trans[i] / 32);
            }
            
            printf("Begin: Iterations %d\n", iterations);
            iterStart = clock();
            
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
    clock_t iterEnd = clock();
    printf("End  : Iterations %d\n", iterations);
    double itertime = ((double) (iterEnd - iterStart)) / CLOCKS_PER_SEC;
    printf("Elapsed time for iterations: %f\n\n", itertime);


    // Result is copied to f from device 0, actually now all devices hold the same result
    cudaSetDevice(0);
    cudaMemcpy(result_f, cuda_f[0], sizeof(float)*cols, cudaMemcpyDeviceToHost);
    float sum = 0;
    for(int i = 0; i < cols; i++)
        sum += result_f[i];
    printf("\nSum f: %f\n\n", sum);

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
    

    clock_t end = clock();
    double totaltime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time totally       : %f\n\n", totaltime);
}


void mlem_test(     int *csr_Rows, int *csr_Cols, float *csr_Vals, 
                    int *csr_Rows_Trans, int *csr_Cols_Trans, float *csr_Vals_Trans, 
                    int *g, float *norm, float *f, float *result_f, int rows, int cols, int nnzs, int using_trans){

    int device = 0, scan_result;
    printf("\nUse which device? (0: Quadro P6000   1: Tesla K20c): ");
    scan_result = scanf("%d", &device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);	
    cudaSetDevice(device);
    printf("\nRunning test MLEM on CUDA device %d (%s)\n\n", device, prop.name);

    clock_t start = clock();
    printf("Begin: Initialization\n");
    clock_t initStart = clock();

    // device variables
    int *cuda_Rows, *cuda_Cols, *cuda_Rows_Trans, *cuda_Cols_Trans, *cuda_g;
    float *cuda_Vals, *cuda_Vals_Trans, *cuda_norm, *cuda_bwproj, *cuda_temp, *cuda_f;

    // allocate device storage
    cudaMalloc((void**)&cuda_Rows, sizeof(int)*(rows + 1));
    cudaMalloc((void**)&cuda_Cols, sizeof(int)*nnzs);
    cudaMalloc((void**)&cuda_Vals, sizeof(float)*nnzs);
    cudaMalloc((void**)&cuda_f, sizeof(float)*cols);
    cudaMalloc((void**)&cuda_g, sizeof(int)*rows);
    cudaMalloc((void**)&cuda_norm, sizeof(float)*cols);
    cudaMalloc((void**)&cuda_bwproj, sizeof(float)*cols);
    cudaMalloc((void**)&cuda_temp, sizeof(float)*rows);
    if(using_trans == 0){
        cudaMalloc((void**)&cuda_Rows_Trans, sizeof(int)*(cols + 1));
        cudaMalloc((void**)&cuda_Cols_Trans, sizeof(int)*nnzs);
        cudaMalloc((void**)&cuda_Vals_Trans, sizeof(float)*nnzs);
    }

    // value initialization
    cudaMemcpy(cuda_Rows, csr_Rows, sizeof(int)*(rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_Cols, csr_Cols, sizeof(int)* nnzs, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_Vals, csr_Vals, sizeof(float)* nnzs, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_g, g, sizeof(int)* rows, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_norm, norm, sizeof(float)* cols, cudaMemcpyHostToDevice);
    cudaMemset(cuda_bwproj, 0, sizeof(float)*cols);
    cudaMemset(cuda_temp, 0, sizeof(float)*rows);
    // cudaMemcpy(cuda_f, f, sizeof(float)* cols, cudaMemcpyHostToDevice);
    if(using_trans == 0){
        cudaMemcpy(cuda_Rows_Trans, csr_Rows_Trans, sizeof(int)*(cols + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Cols_Trans, csr_Cols_Trans, sizeof(int)* nnzs, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Vals_Trans, csr_Vals_Trans, sizeof(float)* nnzs, cudaMemcpyHostToDevice);
    }
    
    clock_t initEnd = clock();
    printf("End  : Initialization\n");
    double initTime = ((double) (initEnd - initStart)) / CLOCKS_PER_SEC;
    printf("Elapsed time for initialization: %f\n", initTime);

    // Determine grid size and section size (block size is set to 1024 by default)
    int blocksize = 1024;
    int gridsize_correl = ceil((double)rows / blocksize);
    int gridsize_update = ceil((double)cols / blocksize);

    
    int once_more = 0;
    int iterations = 10;
    // 0: CSRMV   1:coalesced brutal warp    2: brutal   3: coalesced CSRMV   4: coalesced brutal block
    int matrix_vector_mul = 0;    
    
    while(once_more == 0){
        // reset everytime the program runs
        cudaMemcpy(cuda_f, f, sizeof(float)* cols, cudaMemcpyHostToDevice);
        cudaMemset(cuda_temp,   0, sizeof(float)*rows);
        cudaMemset(cuda_bwproj, 0, sizeof(float)*cols);
        
        
        printf("\nUse which kind of matrix-vector multiplication? (0: CSRMV   1: coalesced brutal warp   2: brutal   3: coalesced CSRMV   4: coalesced brutal block !!! if using no transposed matrix, please give 1): ");
        scan_result = scanf("%d", &matrix_vector_mul);
        
        printf("\nIteration times: ");
        scan_result = scanf("%d", &iterations);
        
        clock_t iterStart;
        
        switch(matrix_vector_mul){
            case 0: { // case 0: CSRMV
                int items_fwproj = rows + nnzs;
                int items_bwproj = cols + nnzs;
                int gridsize_fwproj = ceil(sqrt((double)items_fwproj / blocksize) * 60); 
                int gridsize_bwproj = ceil(sqrt((double)items_bwproj / blocksize) * 15);
                int secsize_fwproj = ceil((double)items_fwproj / (blocksize * gridsize_fwproj));
                int secsize_bwproj = ceil((double)items_bwproj / (blocksize * gridsize_bwproj));
                printf("    rows: %d   cols: %d   nnzs: %d\n", rows, cols, nnzs);
                
                printf("    default grid size forward projection   : %d\n", gridsize_fwproj);
                printf("    default section size forward projection: %d\n", secsize_fwproj);
                printf("    please give section size forward projection: ");
                scan_result = scanf("%d", &secsize_fwproj);
                gridsize_fwproj = ceil((double)items_fwproj / (blocksize * secsize_fwproj));
                printf("    new grid size forward projection: %d\n\n", gridsize_fwproj);

                printf("    default grid size backward projection   : %d\n", gridsize_bwproj);
                printf("    default section size backward projection: %d\n", secsize_bwproj);
                printf("    please give section size backward projection: ");
                scan_result = scanf("%d", &secsize_bwproj);
                gridsize_bwproj = ceil((double)items_bwproj / (blocksize * secsize_bwproj));
                printf("    new grid size borward projection: %d\n\n", gridsize_bwproj);
                
                printf("\nBegin: Iterations %d\n", iterations);
                iterStart = clock();
                for(int i = 0; i < iterations; i++){
                    calcFwProj <<< gridsize_fwproj, blocksize >>> (cuda_Rows, cuda_Cols, cuda_Vals, cuda_f, cuda_temp, secsize_fwproj, rows, nnzs);
                    
                    calcCorrel <<< gridsize_correl, blocksize >>> (cuda_g, cuda_temp, rows);
            
                    calcBwProj <<< gridsize_bwproj, blocksize >>> (cuda_Rows_Trans, cuda_Cols_Trans, cuda_Vals_Trans, cuda_temp, cuda_bwproj, secsize_bwproj, cols, nnzs);
                    
                    calcUpdateInPlace <<< gridsize_update, blocksize >>> (cuda_f, cuda_norm, cuda_bwproj, cols);
            
                    cudaMemset(cuda_temp,   0, sizeof(float)*rows);
                    cudaMemset(cuda_bwproj, 0, sizeof(float)*cols);     
                }
            } break;

            case 1: { // case 1: coalesced brutal warp
                int gridsize_fwproj = ceil((double)rows / 32);
                int gridsize_bwproj = ceil((double)cols / 32);
                int gridsize_bwproj_none_trans = gridsize_fwproj;
                printf("\nBegin: Iterations %d\n", iterations);
                iterStart = clock();
                for(int i = 0; i < iterations; i++){
                    calcFwProj_coalesced_brutal_warp <<< gridsize_fwproj, blocksize >>> (cuda_Rows, cuda_Cols, cuda_Vals, cuda_f, cuda_temp, rows);
            
                    calcCorrel <<< gridsize_correl, blocksize >>> (cuda_g, cuda_temp, rows);

                    if(using_trans == 0)
                        calcBwProj_coalesced_brutal_warp <<< gridsize_bwproj, blocksize >>> (cuda_Rows_Trans, cuda_Cols_Trans, cuda_Vals_Trans, cuda_temp, cuda_bwproj, cols);
                    else
                        calcBwProj_none_trans <<< gridsize_bwproj_none_trans, blocksize >>> (cuda_Rows, cuda_Cols, cuda_Vals, cuda_temp, cuda_bwproj, rows);
            
                    calcUpdateInPlace <<< gridsize_update, blocksize >>> (cuda_f, cuda_norm, cuda_bwproj, cols);

                    cudaMemset(cuda_temp,   0, sizeof(float)*rows);
                    cudaMemset(cuda_bwproj, 0, sizeof(float)*cols);     
                } 
            } break;

            case 2: { //case 2: brutal
                int gridsize_fwproj = gridsize_correl;
                int gridsize_bwproj = gridsize_update;
                printf("\nBegin: Iterations %d\n", iterations);
                iterStart = clock();
                for(int i = 0; i < iterations; i++){
                    calcFwProj_brutal <<< gridsize_fwproj, blocksize >>> (cuda_Rows, cuda_Cols, cuda_Vals, cuda_f, cuda_temp, rows);
            
                    calcCorrel <<< gridsize_correl, blocksize >>> (cuda_g, cuda_temp, rows);
            
                    calcBwProj_brutal <<< gridsize_bwproj, blocksize >>> (cuda_Rows_Trans, cuda_Cols_Trans, cuda_Vals_Trans, cuda_temp, cuda_bwproj, cols);
            
                    calcUpdateInPlace <<< gridsize_update, blocksize >>> (cuda_f, cuda_norm, cuda_bwproj, cols);
                } 
            } break;

            case 3: { // case 4: coalesced CSRMV
                int items_fwproj = rows + nnzs;
                int items_bwproj = cols + nnzs;
                int gridsize_fwproj = ceil((double)items_fwproj / blocksize); 
                int gridsize_bwproj = ceil((double)items_bwproj / blocksize);
                printf("\nBegin: Iterations %d\n", iterations);
                iterStart = clock();
                for(int i = 0; i < iterations; i++){
                    calcFwProj_coalesced <<< gridsize_fwproj, blocksize >>> (cuda_Rows, cuda_Cols, cuda_Vals, cuda_f, cuda_temp, blocksize, rows, nnzs);
                    
                    calcCorrel <<< gridsize_correl, blocksize >>> (cuda_g, cuda_temp, rows);
            
                    calcBwProj_coalesced <<< gridsize_bwproj, blocksize >>> (cuda_Rows_Trans, cuda_Cols_Trans, cuda_Vals_Trans, cuda_temp, cuda_bwproj, blocksize, cols, nnzs);
                    
                    calcUpdateInPlace <<< gridsize_update, blocksize >>> (cuda_f, cuda_norm, cuda_bwproj, cols);
            
                    cudaMemset(cuda_temp,   0, sizeof(float)*rows);
                    cudaMemset(cuda_bwproj, 0, sizeof(float)*cols);     
                }
            } break;
            
            case 4: { // case 4: coalesced brutal block
                printf("\nBegin: Iterations %d\n", iterations);
                iterStart = clock();
                for(int i = 0; i < iterations; i++){
                    calcFwProj_coalesced_brutal_block <<< rows, blocksize >>> (cuda_Rows, cuda_Cols, cuda_Vals, cuda_f, cuda_temp);
            
                    calcCorrel <<< gridsize_correl, blocksize >>> (cuda_g, cuda_temp, rows);
            
                    calcBwProj_coalesced_brutal_block <<< cols, blocksize >>> (cuda_Rows_Trans, cuda_Cols_Trans, cuda_Vals_Trans, cuda_temp, cuda_bwproj);
            
                    calcUpdateInPlace <<< gridsize_update, blocksize >>> (cuda_f, cuda_norm, cuda_bwproj, cols);
                } 
            } break;

            default: break;
        }
            
        cudaDeviceSynchronize();
        clock_t iterEnd = clock();
        printf("End  : Iterations %d\n", iterations);
        double itertime = ((double) (iterEnd - iterStart)) / CLOCKS_PER_SEC;
        printf("Elapsed time for iterations: %f\n\n", itertime);


        // copy result from GPU to CPU and sum up all elements in the solution f
        cudaMemcpy(result_f, cuda_f, sizeof(float)*cols, cudaMemcpyDeviceToHost);
        float sum = 0;
        for(int i = 0; i < cols; i++)
            sum += result_f[i];
        printf("\nSum f: %f\n\n", sum);


        printf("Run program once more? (0: yes   1: no): ");
        scan_result = scanf("%d", &once_more);
        printf("\n");
    }


    // free all memory
    if(cuda_Rows) cudaFree(cuda_Rows);
    if(cuda_Cols) cudaFree(cuda_Cols);
    if(cuda_Vals) cudaFree(cuda_Vals);
    if(cuda_Rows_Trans) cudaFree(cuda_Rows_Trans);
    if(cuda_Cols_Trans) cudaFree(cuda_Cols_Trans);
    if(cuda_Vals_Trans) cudaFree(cuda_Vals_Trans);
    if(cuda_g) cudaFree(cuda_g);
    if(cuda_norm) cudaFree(cuda_norm);
    if(cuda_f) cudaFree(cuda_f);
    if(cuda_bwproj) cudaFree(cuda_bwproj);
    if(cuda_temp) cudaFree(cuda_temp);


    clock_t end = clock();
    double totaltime = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time totally       : %f\n\n", totaltime);
}


int main(){
    int scan_result;
    // 0: test mlem    1: nccl mlem
    int MLEM_Version = 0;
    // 0: using small matrix   1: using big matrix
    int small = 0;
    // 0: using transposed matrix   1: using no transposed matrix
    int using_trans = 0;
    printf("\nMLEM version (0: test version   1: nccl version): ");
    scan_result = scanf("%d", &MLEM_Version);
    printf("\nUse which matrix? (0: small matrix   1: big matrix): ");
    scan_result = scanf("%d", &small);
    printf("\nUse transposed matrix? (0: yes   1: no): ");
    scan_result = scanf("%d", &using_trans);
    printf("\n");

    // host variables
    int *csr_Rows, *csr_Cols, *csr_Rows_Trans, *csr_Cols_Trans, *g, rows, cols, nnzs, sum_g = 0;
    float *csr_Vals, *csr_Vals_Trans, *f, *result_f, *norm, sum_norm = 0.0f;

    // duplicate host variables, only used for nccl mlem because csr_Rows and csr_Rows_Trans are modified there
    int *csr_Rows_copy, *csr_Rows_Trans_copy;


    // read matrix
    printf("Begin: Read Matrix\n");
    std::string matrixPath = small == 0? "/scratch/pet/madpet2.p016.csr4.small" : "/scratch/pet/madpet2.p016.csr4";
    Csr4Matrix matrix(matrixPath);
    printf("End  : Read Matrix\n\n");
    printf("Begin: Create CSR Format for Matrix\n");
    clock_t start = clock();
    rows = matrix.rows();
    cols = matrix.columns();
    nnzs = matrix.elements();
    printf("    The matrix contains %d rows, %d cols, %d nnzs\n", rows, cols, nnzs);
    matrix.mapRows(0, rows);    
    csr_Rows = (int*)malloc(sizeof(int) * (rows + 1));
    csr_Cols = (int*)malloc(sizeof(int) * nnzs);
    csr_Vals = (float*)malloc(sizeof(float) * nnzs);
    csr_format_for_cuda(matrix, csr_Vals, csr_Rows, csr_Cols);
    Vector<float> norm_helper(cols, 0.0);
    calcColumnSums(matrix, norm_helper);
    norm = norm_helper.ptr();
    // TODO: calculate sum_norm using gpu
    for(int i = 0; i < cols; i++)
        sum_norm += norm[i];
    clock_t end = clock();
    printf("End  : Create CSR Format for Matrix\n");
    double elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time for creating CSR: %f\n\n", elapsed);
    
    if(MLEM_Version == 1){
        csr_Rows_copy = (int*)malloc(sizeof(int) * (rows + 1));
        memcpy(csr_Rows_copy, csr_Rows, sizeof(int) * (rows + 1));
    }

    // read image
    printf("Begin: Read Image\n");
    start = clock();
    std::string imagePath = small == 0? "/scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino.small" : "/scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino";
    Vector<int> image(imagePath);
    g = image.ptr();
    // TODO: calculate sum_g using gpu
    for(int i = 0; i < rows; i++)
        sum_g += g[i];
    end = clock();
    printf("End  : Read Image\n");
    elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time for reading image: %f\n\n", elapsed);

    // calculate initial value
    float init = sum_g / sum_norm;
    printf("Sum of norms: %f\n", sum_norm);
    printf("Sum of g    : %d\n", sum_g);
    printf("Initial f   : %f\n\n", init);
    f = (float*)malloc(sizeof(float)*cols);
    result_f = (float*)malloc(sizeof(float)*cols);
    for(int i = 0; i < cols; i++)
        f[i] = init;
    

    // !!!!!!!!!!!!!!!!!!!!!!
    /*
    if(small != 0){
        rows = fiveSixth(csr_Rows, nnzs, rows);
        nnzs = csr_Rows[rows];
        printf("\nNow rows is %d, nnzs is %d\n", rows, nnzs);
    }*/


    if(using_trans == 0){
        // transpose matrix
        printf("Begin: Transpose Matrix\n");
        clock_t start_Trans = clock();
        // transpose matrix using GPU
        // transposeCSR(cuda_Rows, cuda_Cols, cuda_Vals, cuda_Rows_Trans, cuda_Cols_Trans, cuda_Vals_Trans, rows, cols, nnzs);
        
        // transpose matrix using CPU
        csr_Rows_Trans = (int*) calloc (cols+1,sizeof(int));
        csr_Cols_Trans = (int*) calloc (nnzs,sizeof(int));
        csr_Vals_Trans = (float*) calloc (nnzs,sizeof(float));
        sptrans_scanTrans<int, float>(rows, cols, nnzs, csr_Rows, csr_Cols, csr_Vals, csr_Cols_Trans, csr_Rows_Trans, csr_Vals_Trans);
        clock_t end_Trans = clock();
        printf("End  : Transpose Matrix\n");
        elapsed = ((double) (end_Trans - start_Trans)) / CLOCKS_PER_SEC;
        printf("Elapsed time for transposing matrix: %f\n\n", elapsed);

        if(MLEM_Version == 1){
            csr_Rows_Trans_copy = (int*)malloc(sizeof(int) * (cols + 1));
            memcpy(csr_Rows_Trans_copy, csr_Rows_Trans, sizeof(int) * (cols + 1));
        }
    }

    
    // run mlem algorithm matrix
    printf("\n****************************************************************\n");
    printf("Begin: Run MLEM function\n");
    switch(MLEM_Version){
        case 0: mlem_test(csr_Rows, csr_Cols, csr_Vals, csr_Rows_Trans, csr_Cols_Trans, csr_Vals_Trans, g, norm, f, result_f, rows, cols, nnzs, using_trans); break;
        case 1: {
            int once_more = 0;
            while(once_more == 0){

                if(using_trans == 0)
                    mlem_nccl(csr_Rows, csr_Cols, csr_Vals, csr_Rows_Trans, csr_Cols_Trans, csr_Vals_Trans, g, norm, f, result_f, rows, cols, nnzs); 
                else
                    mlem_nccl_none_trans(csr_Rows, csr_Cols, csr_Vals, g, norm, f, result_f, rows, cols, nnzs);

                printf("Run program once more? (0: yes   1: no): ");
                scan_result = scanf("%d", &once_more);
                printf("\n");

                if(once_more == 0){
                    memcpy(csr_Rows, csr_Rows_copy, sizeof(int) * (rows + 1));

                    if(using_trans == 0)
                        memcpy(csr_Rows_Trans, csr_Rows_Trans_copy, sizeof(int) * (cols + 1));
                }
            }
        } break;
        default: break;
    }
    printf("End  : Run MLEM function\n");
    printf("****************************************************************\n");

    
    if (csr_Rows) free(csr_Rows);
    if (csr_Cols) free(csr_Cols);
    if (csr_Vals) free(csr_Vals);
    if(csr_Rows_copy) free(csr_Rows_copy);
    // if (g) free(g);
    // if (norm) free(norm);
    if (f) free(f);
    if(result_f) free(result_f);
    if(using_trans == 0){
        if (csr_Rows_Trans) free(csr_Rows_Trans);
        if (csr_Cols_Trans) free(csr_Cols_Trans);
        if (csr_Vals_Trans) free(csr_Vals_Trans);
        if(csr_Rows_Trans_copy) free(csr_Rows_Trans_copy);
    }

    return 0;
}
