
#ifndef _KERNEL_CUH_ 
#define _KERNEL_CUH_

// MLEM functions using CSR and merge-based CSRMV 

__global__ void calcFwProj(int *csr_Row, int *csr_Col, float *csr_Val, float *f, float *fwproj, int secSize, int rows, int nnzs);
__global__ void calcCorrel(int *g, float *fwproj, int rows);
__global__ void calcBwProj(int *csr_Row_Trans, int *csr_Col_Trans, float *csr_Val_Trans, float *correl, float *bwproj, int secSize, int cols, int nnzs);
__global__ void calcUpdate(float *f, float *norm, float *bwproj, int cols);
__global__ void calcUpdateInPlace(float *f, float *norm, float *bwproj, int cols);

// naive implementation, brutal calculation
__global__ void calcFwProj_naive(int *csr_Row, int *csr_Col, float *csr_Val, float *f, float *fwproj, int rows);
__global__ void calcBwProj_naive(int *csr_Row_Trans, int *csr_Col_Trans, float *csr_Val_Trans, float *correl, float *bwproj, int cols);


// help functions for sparse matrix-vector multiplication using CSR and merge-based CSRMV

__device__ void SpMV_start(int *csr_Row, int *csr_Col, float *csr_Val, float *x, float *result, int secSize, int rows, int nnzs);
__device__ void SpMV_work (int *csr_Row, int *csr_Col, float *csr_Val, float *x, float *result, int secSize, int rows, int nnzs, int i, int j);

__device__ void naive_matrix_vector_mul(int *csr_Row, int *csr_Col, float *csr_Val, float *x, float *result, int rows);

#endif