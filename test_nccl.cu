#include "algorithm"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "cusparse.h"
#include "time.h"
#include "nccl.h"

int main (){
    int devs;
    cudaGetDeviceCount(&devs);
    printf("capable devices: %d\n", devs);

    float **f = (float**)malloc(devs * sizeof(float*));
    float v = 0.0f;
    for(int i = 0; i < devs; i++){
        f[i] = (float*)malloc(5*sizeof(float));
        for(int j = 0; j < 5; j++, v+=1.0f)
            f[i][j] = v;        
    }
    
    int *devices = (int*)malloc(devs * sizeof(int));    

    ncclComm_t *comms = (ncclComm_t*)malloc(devs * sizeof(ncclComm_t));;

    cudaStream_t *streams = (cudaStream_t*)malloc(devs * sizeof(cudaStream_t));

    float **cuda_f1 = (float**)malloc(devs * sizeof(float*));
    float **cuda_f2 = (float**)malloc(devs * sizeof(float*));
    for(int i = 0; i < devs; i++){
        cudaSetDevice(i);
        cudaMalloc((void**)&cuda_f1[i], sizeof(float)*5);
        cudaMalloc((void**)&cuda_f2[i], sizeof(float)*5);
        cudaMemcpy(cuda_f1[i], f[i], sizeof(float)*5, cudaMemcpyHostToDevice);
        cudaStreamCreate(streams+i);
        devices[i] = i;
    }
    
    ncclCommInitAll(comms, devs, devices);

    ncclGroupStart();
    for (int i = 0; i < devs; i++)
        ncclAllReduce((const void*)cuda_f1[i], (void*)cuda_f1[i], 5, ncclFloat, ncclProd, comms[i], streams[i]);
    ncclGroupEnd();

    for (int i = 0; i < devs; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }


    for(int i = 0; i < devs; i++){
        cudaMemcpy(f[i], cuda_f1[i], sizeof(float)*5, cudaMemcpyDeviceToHost);
        printf("values in device %d: ", i);
        for(int j = 0; j < 5; j++)
            printf("%f   ", f[i][j]);
        printf("\n");
    }
    

    for (int i = 0; i < devs; ++i) {
        free(f[i]);
        
        cudaSetDevice(i);
        cudaFree(cuda_f1[i]);
        cudaFree(cuda_f2[i]);
        ncclCommDestroy(comms[i]);
    }
    
    free(f);
    free(comms);
    free(devices);
    free(streams);
    free(cuda_f1);
    free(cuda_f2);
    
    
    return 0;
}