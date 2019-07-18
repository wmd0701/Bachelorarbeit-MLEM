
#ifndef _KERNEL_CUH_ 
#define _KERNEL_CUH_

// MLEM functions using CSR and merge-based CSRMV 
__global__ void calcFwProj_coalesced_brutal_warp(unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *f, float *fwproj, unsigned int rows);
__global__ void calcCorrel(int *g, float *fwproj, unsigned int rows);
__global__ void calcBwProj_none_trans(unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *correl, float *bwproj, unsigned int rows);
__global__ void calcUpdate(float *f, float *norm, float *bwproj, unsigned int cols);



// coalesced brutal version matrix-vector multiplication, totally same with the master thesis from last year
// each warp calculates one row in matrix multiplied with the vector
__device__ void mat_vec_mul_coalesced_brutal_warp(unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *x, float *result, unsigned int rows);

// backward projection help function for using none transposed matrix
__device__ void trans_mat_vec_mul_warp(unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *x, float *result, unsigned int rows);

#endif