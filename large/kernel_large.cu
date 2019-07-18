#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"


__global__ void calcFwProj_coalesced_brutal_warp (unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *f, float *fwproj, unsigned int rows){
	mat_vec_mul_coalesced_brutal_warp (csr_Rows, csr_Cols, csr_Vals, f, fwproj, rows);
}

/*
	brief: calculate correlation, output saved in fwproj in-place
	@param g:			measurement array
	@param fwproj:		result of forward projection / output array
	@param rows:			number of rows (equals to length of row array - 1)
*/
__global__ void calcCorrel(int *g, float *fwproj, unsigned int rows) {
	
	// !!! gridsize x blocksize >= rows
	
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < rows) 
		if(fwproj[index] != 0.0f)
			fwproj[index] =  g[index] / fwproj[index];
}


/*
	brief: calculate update, output saved in bwproj, for mlem nccl
	@param f:			input array
	@param norm:		norm array
	@param bwproj:		result of backward projection / output array
	@param cols:		number of columns of original matrix
*/
__global__ void calcUpdate(float *f, float *norm, float *bwproj, unsigned int cols) {
	
	// !!!gridsize x blocksize >= cols

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < cols) {
		if(norm[index] == 0.0f)
			bwproj[index] = f[index] * bwproj[index];
		else
			bwproj[index] = f[index] * bwproj[index] / norm[index];
	}
}

__global__ void calcBwProj_none_trans(unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *correl, float *bwproj, unsigned int rows){
	trans_mat_vec_mul_warp(csr_Rows, csr_Cols, csr_Vals, correl, bwproj, rows);
}


__device__ void mat_vec_mul_coalesced_brutal_warp (unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *x, float *result, unsigned int rows){
	__shared__ float values[1024];

	unsigned int WARP_SIZE = 32;

	unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp

	unsigned int warp_id = thread_id / WARP_SIZE; // global warp index
	// total number of active warps
	unsigned int num_warps = (blockDim.x / WARP_SIZE) * gridDim.x;
	// one warp per row
	for (unsigned int row = warp_id; row < rows ; row += num_warps){
		unsigned int row_start = csr_Rows[row];
		unsigned int row_end = csr_Rows[row + 1];
		
		// compute running sum per thread
		values[threadIdx.x] = 0.0;
		
		for (unsigned int jj = row_start + thread_lane ; jj < row_end ; jj += WARP_SIZE)
			values[threadIdx.x] += csr_Vals[jj] * x[csr_Cols[jj]];

		// first thread writes the result
		if (thread_lane == 0){
			for (unsigned int i = 1 ; i < WARP_SIZE ; i++)
				values[threadIdx.x] += values[threadIdx.x + i];
			
			atomicAdd(result + row, values[threadIdx.x]);
		}

		__syncthreads();
	}
}


__device__ void trans_mat_vec_mul_warp(unsigned int *csr_Rows, unsigned int *csr_Cols, float *csr_Vals, float *x, float *result, unsigned int rows){

	unsigned int WARP_SIZE = 32;

	unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp

	unsigned int warp_id = thread_id / WARP_SIZE; // global warp index
	// total number of active warps
	unsigned int num_warps = (blockDim.x / WARP_SIZE) * gridDim.x;
	for(unsigned int row = warp_id; row < rows ; row += num_warps){
		unsigned int row_start = csr_Rows[row];
		unsigned int row_end   = csr_Rows[row + 1];
		for (unsigned int i= row_start + thread_lane; i < row_end; i += WARP_SIZE)
			atomicAdd(&result[csr_Cols[i]], csr_Vals[i] * x[row]);
	}
}
