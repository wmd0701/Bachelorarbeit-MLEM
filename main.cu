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

// #define TransposeMatrixUsingCPU true
#define Iterations 300

void csr_format_for_cuda(const Csr4Matrix& matrix, float* csrVal, int* csrRowInd, int* csrColInd){   
    int index = 0;
    csrRowInd[index] = 0;
// pragma omp parallel for schedule (static)
    for (int row = 0; row < matrix.rows(); ++row) {
        csrRowInd[row + 1] = csrRowInd[row] + (int)matrix.elementsInRow(row);
	
        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),[&](const RowElement<float>& e){ 
            csrVal[index] = e.value();
            csrColInd[index] = (int)e.column() ;
            index = index + 1; }
        );
    }
}

void calcColumnSums(const Csr4Matrix& matrix, Vector<float>& norm)
{
    assert(matrix.columns() == norm.size());

    std::fill(norm.ptr(), norm.ptr() + norm.size(), 0.0);
    matrix.mapRows(0, matrix.rows());

  // pragma omp parallel for schedule (static)
    for (uint32_t row=0; row<matrix.rows(); ++row) {
        std::for_each(matrix.beginRow2(row), matrix.endRow2(row),
                      [&](const RowElement<float>& e){ norm[e.column()] += e.value(); });
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
        if(csr_Rows[i] > halfnnzs)
            break;
    return i;
}

void mlem(  int *csr_Rows, float *csr_Vals, int *csr_Cols, 
            int *csr_Rows_Trans, int *csr_Cols_Trans, float *csr_Vals_Trans, 
            int *g, float *norm, float *f, int rows, int cols, int nnzs){
    clock_t start = clock();
    
    // halve the matrix
    int halfrows1 = halfMatrix(csr_Rows, nnzs, rows);
    int halfrows2 = rows - halfrows1;
    int halfnnzs1 = csr_Rows[halfrows1];
    int halfnnzs2 = nnzs - halfnnzs1;
    int offset = csr_Rows[halfrows1];
    printf("First  half matrix contains %d rows and %d nnzs\n", halfrows1, halfnnzs1);
    printf("Second half matrix contains %d rows and %d nnzs\n", halfrows2, halfnnzs2);
    // adjust row array for the second half matrix
    // TODO: accelerate this adjustment with GPU
    for(int i = halfrows1+1; i <= rows; i++)
        csr_Rows[i] -= offset;

    // halve the transposed matrix
    int halfrows1_trans = halfMatrix(csr_Rows_Trans, nnzs, cols);
    int halfrows2_trans = cols - halfrows1_trans;
    int halfnnzs1_trans = csr_Rows_Trans[halfrows1_trans];
    int halfnnzs2_trans = nnzs - halfnnzs1_trans;
    int offset_trans = csr_Rows_Trans[halfrows1_trans];
    printf("First  half transposed matrix contains %d rows and %d nnzs\n", halfrows1_trans, halfnnzs1_trans);
    printf("Second half transposed matrix contains %d rows and %d nnzs\n", halfrows2_trans, halfnnzs2_trans);
    // adjust row array for the second half matrix
    // TODO: accelerate this adjustment with GPU
    for(int i = halfrows1_trans+1; i <= cols; i++)
        csr_Rows_Trans[i] -= offset_trans;


    // device variables
    int *cuda_Rows, *cuda_Cols, *cuda_Rows_Trans, *cuda_Cols_Trans, *cuda_g;
    float *cuda_Vals, *cuda_Vals_Trans, *cuda_norm, *cuda_bwproj, *cuda_temp, *cuda_f;


    // allocate device storage
    printf("    Begin: Allocate GPU Storage\n");
    int rows_init = halfrows1 > halfrows2 ? halfrows1 : halfrows2;
    int nnzs_init = halfnnzs1 > halfnnzs2 ? halfnnzs1 : halfnnzs2;
    int rows_init_trans = halfrows1_trans > halfrows2_trans ? halfrows1_trans : halfrows2_trans;
    int nnzs_init_trans = halfnnzs1_trans > halfnnzs2_trans ? halfnnzs1_trans : halfnnzs2_trans;

    cudaMalloc((void**)&cuda_Rows, sizeof(int)*(rows_init + 1));
    cudaMalloc((void**)&cuda_Cols, sizeof(int)*nnzs_init);
    cudaMalloc((void**)&cuda_Vals, sizeof(float)*nnzs_init);
    cudaMalloc((void**)&cuda_Rows_Trans, sizeof(int)*(rows_init_trans + 1));
    cudaMalloc((void**)&cuda_Cols_Trans, sizeof(int)*nnzs_init_trans);
    cudaMalloc((void**)&cuda_Vals_Trans, sizeof(float)*nnzs_init_trans);
    cudaMalloc((void**)&cuda_f, sizeof(float)*cols);
    cudaMalloc((void**)&cuda_g, sizeof(int)*rows);
    cudaMalloc((void**)&cuda_norm, sizeof(float)*cols);
    cudaMalloc((void**)&cuda_bwproj, sizeof(float)*cols);
    cudaMalloc((void**)&cuda_temp, sizeof(float)*rows);
    printf("    End  : Allocate GPU Storage\n");

    // value initialization
    printf("    Begin: GPU Storage Initialization\n");
    cudaMemcpy(cuda_g, g, sizeof(int)* rows, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_norm, norm, sizeof(float)* cols, cudaMemcpyHostToDevice);
    cudaMemset(cuda_bwproj, 0, sizeof(float)*cols);
    cudaMemset(cuda_temp, 0, sizeof(float)*rows);
    // cudaMemset(cuda_f, init, sizeof(float)*cols);
    // cuMemsetD32(cuda_f, __float_as_int(init), cols);
    cudaMemcpy(cuda_f, f, sizeof(float)* cols, cudaMemcpyHostToDevice);
    cudaMemset(cuda_Rows_Trans, 0, sizeof(int)*(cols+1));
    cudaMemset(cuda_Cols_Trans, 0, sizeof(int)*nnzs);
    cudaMemset(cuda_Vals_Trans, 0, sizeof(float)*nnzs);
    printf("    End  : GPU Storage Initialization\n");

    
    // Determine grid size and section size (block size is set to 1024 by default)
    int blocksize = 1024;
    int gridsize_correl = ceil((double)rows / blocksize);
    int gridsize_update = ceil((double)cols / blocksize);
    int items_fwproj1 = halfrows1 + halfnnzs1;
    int items_fwproj2 = halfrows2 + halfnnzs1;
    int items_bwproj1 = halfrows1_trans + halfnnzs1_trans;
    int items_bwproj2 = halfrows2_trans + halfnnzs2_trans;
    int gridsize_fwproj1 = ceil(sqrt((double)items_fwproj1 / blocksize));
    int gridsize_fwproj2 = ceil(sqrt((double)items_fwproj2 / blocksize));
    int gridsize_bwproj1 = ceil(sqrt((double)items_bwproj1 / blocksize));
    int gridsize_bwproj2 = ceil(sqrt((double)items_bwproj2 / blocksize));
    int secsize_fwproj1 = ceil((double)items_fwproj1 / (blocksize * gridsize_fwproj1));
    int secsize_fwproj2 = ceil((double)items_fwproj2 / (blocksize * gridsize_fwproj2));
    int secsize_bwproj1 = ceil((double)items_bwproj1 / (blocksize * gridsize_bwproj1));
    int secsize_bwproj2 = ceil((double)items_bwproj2 / (blocksize * gridsize_bwproj2));

    
    // iterations
    printf("    Begin: Iterations %d\n", Iterations);
    clock_t startIter = clock();
    for(int i = 0; i < Iterations; i++){
        // forward projection for first half matrix
        csr_Rows[halfrows1] = offset;
        cudaMemcpy(cuda_Rows, csr_Rows, sizeof(int)*(halfrows1 + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Cols, csr_Cols, sizeof(int)* halfnnzs1, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Vals, csr_Vals, sizeof(float)* halfnnzs1, cudaMemcpyHostToDevice);
        calcFwProj <<< gridsize_fwproj1, blocksize >>> (cuda_Rows, cuda_Vals, cuda_Cols, cuda_f, cuda_temp, secsize_fwproj1, halfrows1, halfnnzs1);

        // forward projection for second half matrix
        csr_Rows[halfrows1] = 0;
        cudaMemcpy(cuda_Rows, csr_Rows+halfrows1, sizeof(int)*(halfrows2 + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Cols, csr_Cols+halfnnzs1, sizeof(int)* halfnnzs2, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Vals, csr_Vals+halfnnzs1, sizeof(float)* halfnnzs2, cudaMemcpyHostToDevice);
        calcFwProj <<< gridsize_fwproj2, blocksize >>> (cuda_Rows, cuda_Vals, cuda_Cols, cuda_f, cuda_temp+halfrows1, secsize_fwproj2, halfrows2, halfnnzs2);
        
        // correlation
        calcCorrel <<< gridsize_correl, blocksize >>> (cuda_g, cuda_temp, rows);

        // backward projection for first half transposed matrix
        csr_Rows_Trans[halfrows1_trans] = offset_trans;
        cudaMemcpy(cuda_Rows_Trans, csr_Rows_Trans, sizeof(int)*(halfrows1_trans + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Cols_Trans, csr_Cols_Trans, sizeof(int)* halfnnzs1_trans, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Vals_Trans, csr_Vals_Trans, sizeof(float)* halfnnzs1_trans, cudaMemcpyHostToDevice);
        calcBkProj <<< gridsize_bwproj1, blocksize >>> (cuda_Rows_Trans, cuda_Vals_Trans, cuda_Cols_Trans, cuda_temp, cuda_bwproj, secsize_bwproj1, halfrows1_trans, halfnnzs1_trans);

        // backward projection for second half transposed matrix
        csr_Rows_Trans[halfrows1_trans] = 0;
        cudaMemcpy(cuda_Rows_Trans, csr_Rows_Trans+halfrows1_trans, sizeof(int)*(halfrows2_trans + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Cols_Trans, csr_Cols_Trans+halfnnzs1_trans, sizeof(int)* halfnnzs2_trans, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_Vals_Trans, csr_Vals_Trans+halfnnzs1_trans, sizeof(float)* halfnnzs2_trans, cudaMemcpyHostToDevice);
        calcBkProj <<< gridsize_bwproj2, blocksize >>> (cuda_Rows_Trans, cuda_Vals_Trans, cuda_Cols_Trans, cuda_temp, cuda_bwproj+halfrows1_trans, secsize_bwproj2, halfrows2_trans, halfnnzs2_trans);
        
        // update
        calcUpdate <<< gridsize_update, blocksize >>> (cuda_f, cuda_norm, cuda_bwproj, cols);
        
        // clear temp vector
        clearTemp  <<< gridsize_correl, blocksize >>> (cuda_temp, rows);
    }
    clock_t endIter = clock();
    printf("    End  : Iterations %d\n\n", Iterations);
    
    // Result is copied to f
    cudaMemcpy(f, cuda_f, sizeof(float)*cols, cudaMemcpyDeviceToHost);

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
    double itertime = ((double) (endIter - startIter)) / CLOCKS_PER_SEC;
    printf("    Time for the whole MLEM function: %f\n", totaltime);
    printf("    Time for the MLEM iterations: %f\n\n", itertime);
}


int main(){

    // host variables
    int *csr_Rows, *csr_Cols, *g, rows, cols, nnzs, sum_g = 0;
    float *csr_Vals, *norm, sum_norm = 0.0f;


    // read matrix
    printf("Begin: Read Matrix\n");
    Csr4Matrix matrix("/scratch/pet/madpet2.p016.csr4.small");
    printf("End  : Read Matrix\n");
    printf("Begin: Create CSR Format for Matrix\n");
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
    // TODO: calculate sum_norm using gpu
    for(int i = 0; i < cols; i++)
        sum_norm += norm[i];
    printf("End  : Create CSR Format for Matrix\n");
    

    // read image
    printf("Begin: Read Image\n");
    Vector<int> image("/scratch/pet/Trues_Derenzo_GATE_rot_sm_200k.LMsino.small");
    g = image.ptr();
    // TODO: calculate sum_g using gpu
    for(int i = 0; i < rows; i++)
        sum_g += g[i];
    printf("End  : Read Image\n\n");
    

    float init = sum_g / sum_norm;
    printf("Sum of norms: %f\n", sum_norm);
    printf("Sum of g    : %d\n", sum_g);
    printf("Initial f   : %f\n\n", init);


    // transpose matrix
    printf("Begin: Transpose Matrix\n");
    // transpose matrix using GPU
    // transposeCSR(cuda_Rows, cuda_Cols, cuda_Vals, cuda_Rows_Trans, cuda_Cols_Trans, cuda_Vals_Trans, rows, cols, nnzs);
    
    // transpose matrix using CPU
    int *csr_Rows_Trans = (int*) calloc (cols+1,sizeof(int));
    int *csr_Cols_Trans = (int*) calloc (nnzs,sizeof(int));
    float *csr_Vals_Trans = (float*) calloc (nnzs,sizeof(float));
    sptrans_scanTrans<int, float>(rows, cols, nnzs, csr_Rows, csr_Cols, csr_Vals, csr_Cols_Trans, csr_Rows_Trans, csr_Vals_Trans);
    printf("End  : Transpose Matrix\n");

    float *f = (float*)malloc(sizeof(float)*cols);
    for(int i = 0; i < cols; i++)
        f[i] = init;
    
    // run mlem algorithm matrix
    printf("\n\n******************************\n");
    printf("Begin: Run MLEM for %d iterations\n", Iterations);
    mlem(csr_Rows, csr_Vals, csr_Cols, csr_Rows_Trans, csr_Cols_Trans, csr_Vals_Trans, g, norm, f, rows, cols, nnzs);
    printf("End  : Run MLEM for %d iterations\n", Iterations);
    printf("******************************\n");

    // sum up all elements in the solution f
    float sum = 0;
    for(int i = 0; i < cols; i++)
        sum += f[i];
    
    printf("\nSum f: %f\n\n", sum);
    
    if (csr_Rows) free(csr_Rows);
    if (csr_Cols) free(csr_Cols);
    if (csr_Vals) free(csr_Vals);
    if (csr_Rows_Trans) free(csr_Rows_Trans);
    if (csr_Cols_Trans) free(csr_Cols_Trans);
    if (csr_Vals_Trans) free(csr_Vals_Trans);
    // if (g) free(g);
    // if (norm) free(norm);
    if (f) free(f);

    return 0;
}
