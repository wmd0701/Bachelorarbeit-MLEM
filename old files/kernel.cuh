
#ifndef _KERNEL_CUH_ 
#define _KERNEL_CUH_


// calculate correlation and calculate update
__global__ void calcCorrel(int *g, float *fwproj, int rows);
__global__ void calcUpdate(float *f, float *norm, float *bwproj, int cols);
__global__ void calcUpdateInPlace(float *f, float *norm, float *bwproj, int cols);

// MLEM functions using CSR and merge-based CSRMV 
__global__ void calcFwProj_merge_based(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *f, float *fwproj, int secSize, int rows, int nnzs);
__global__ void calcBwProj_merge_based(int *csr_Rows_Trans, int *csr_Cols_Trans, float *csr_Vals_Trans, float *correl, float *bwproj, int secSize, int cols, int nnzs);


// forward/backward projection using brutal matrix-vector multiplication
__global__ void calcFwProj_brutal(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *f, float *fwproj, int rows);
__global__ void calcBwProj_brutal(int *csr_Rows_Trans, int *csr_Cols_Trans, float *csr_Vals_Trans, float *correl, float *bwproj, int cols);

/* 
    forward/backward projection using coalesced CSRMV matrix-vector multiplication
    difference: normal forward/backward projection: each thread one section
                coalesced forward/backward projection: each block one section
    secSize is actually 1024
*/
__global__ void calcFwProj_coalesced(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *f, float *fwproj, int secSize, int rows, int nnzs);
__global__ void calcBwProj_coalesced(int *csr_Rows_Trans, int *csr_Cols_Trans, float *csr_Vals_Trans, float *correl, float *bwproj, int secSize, int cols, int nnzs);

// forward/backward projection using coalesced brutal block matrix-vector multiplication
__global__ void calcFwProj_coalesced_brutal_block(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *f, float *fwproj);
__global__ void calcBwProj_coalesced_brutal_block(int *csr_Rows_Trans, int *csr_Cols_Trans, float *csr_Vals_Trans, float *correl, float *bwproj);

// forward/backward projection using coalesced brutal warp matrix-vector multiplication
__global__ void calcFwProj_coalesced_brutal_warp(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *f, float *fwproj, int rows);
__global__ void calcBwProj_coalesced_brutal_warp(int *csr_Rows_Trans, int *csr_Cols_Trans, float *csr_Vals_Trans, float *correl, float *bwproj, int cols);


// backward projection using no transposed matrix
__global__ void calcBwProj_none_trans(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *correl, float *bwproj, int rows);




// matrix-vector multiplication using merge-based CSRMV
__device__ void SpMV_start(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *x, float *result, int secSize, int rows, int nnzs);
__device__ void SpMV_work (int *csr_Rows, int *csr_Cols, float *csr_Vals, float *x, float *result, int secSize, int rows, int nnzs, int i, int j);

// coalesced version matrix-vector multiplication using CSRMV
__device__ void SpMV_start_coalesced(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *x, float *result, int secSize, int rows, int nnzs);
__device__ void SpMV_work_coalesced(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *x, float *result, int rows, int nnzs, int i, int j);

// brutal matrix-vector multiplication
__device__ void mat_vec_mul_brutal(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *x, float *result, int rows);

// coalesced brutal version matrix-vector multiplication, idea basically same with the master thesis from last year
// each block calculates one row in matrix multiplied with the vector
__device__ void mat_vec_mul_coalesced_brutal_block(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *x, float *result);

// coalesced brutal version matrix-vector multiplication, totally same with the master thesis from last year
// each warp calculates one row in matrix multiplied with the vector
__device__ void mat_vec_mul_coalesced_brutal_warp(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *x, float *result, int rows);

// backward projection help function for using none transposed matrix
__device__ void trans_mat_vec_mul_warp(int *csr_Rows, int *csr_Cols, float *csr_Vals, float *x, float *result, int rows);

#endif