
#ifndef _KERNEL_CUH_ 
#define _KERNEL_CUH_

// MLEM functions using CSR and merge-based CSRMV 

__global__ void calcFwProj(int *csr_Row, float *csr_Val, int *csr_Col, float *f, float *fwproj, int secSize, int rows, int nnzs);
__global__ void calcCorrel(int *g, float *fwproj, int rows);
__global__ void calcBkProj(int *csr_Row, float *csr_Val, int *csr_Col, float *correl, float *bwproj, int secSize, int cols, int nnzs);
__global__ void calcUpdate(float *f, float *norm, float *bwproj, int cols);
__global__ void clearTemp(float *temp, int rows);



// help functions for sparse matrix-vector multiplication using CSR and merge-based CSRMV

__device__ void SpMV_start(int *csr_Row, float *csr_Val, int *csr_Col, float *x, float *result, int secSize, int rows, int nnzs);
__device__ void SpMV_work (int *csr_Row, float *csr_Val, int *csr_Col, float *x, float *result, int secSize, int rows, int nnzs, int i, int j);

#endif