# MLEM - Maximum Liklihood Expectation Maximization for Small PET Devices

This is the project for Mengdi' bachelor thesis for the topic: porting MLEM algorithm to heterogeneous systems. The main purpose is to develop a special version of MLEM algorithm working on multiple GPUs, while other parts of the work are done on CPU.

Most of the pre-process work such as loading matrix, transporing matrix, partitioning matrix etc. are done on CPUs.

Others are done on GPUs.

## Files
- Makefile:     makefile
- kernel.cu:    core functions for MLEM algorithm
- kernel.cuh:   cuda header file for kernel.cu
- main.cu:      program main body, including main() function
- sptrans.h:    functions for matrix transposition

## other folders
- bash code:            bach codes to run the program sophisticatedly, output and store information given by nvprof
- helper_files_common:  helper functions for loading matrix and creating CSR format, provided by the Department of Informatics, TUM
- old files:            older versions of implementation

### Compile
```sh
make
```
### Usage
```sh
./test
```

### Parameters
```sh
total number of parameters: 9
argv[1]: path of matrix file
argv[2]: path of image  file
argv[3]: iteration times
argv[4]: number of GPUs to be used
argv[5]: section size for forward  projection in NVIDIA merge-based SpMV
argv[6]: section size for backward projection in NVIDIA merge-based SpMV
argv[7]: whether to use transposed matrix    0: yes   1: no
argv[8]: which SpMV to used     0: NVIDIA merge-based SpMV    1: Coalesced Brutal Warp SpMV
```