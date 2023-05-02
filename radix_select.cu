#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "radix_select.h"

#define UINT32_MAX 4294967295

__device__ unsigned int hash(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

__global__ void randf(float *p, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while(idx < n){
        p[idx] = (float)hash(idx + 1) / UINT32_MAX; 
        idx += blockDim.x * gridDim.x;
    }
}

static const int N = 1 << 30;
static const int topk = 100;
int origin()
{
    printf("start\n");
    typedef float T;
    assert(sizeof(T) == 4); // change this need to change func:randd.
    T *p_d;
    cudaMalloc((void **)&p_d, N * sizeof(T)); 
    T *answer;  
    answer =(T *)malloc(topk * sizeof(T)); 

    randf<<<1024, 1024>>>(p_d, N);
    

    ////////////////////////////////////////////////////////////////////////////
    cudaEvent_t ev1, ev2;
	cudaEventCreate(&ev1);
	cudaEventCreate(&ev2);
    cudaEventRecord(ev1);
    
    radix_select(p_d, N, answer, topk);

    cudaGetLastError();
	cudaEventRecord(ev2);
	cudaDeviceSynchronize();
    float elapse = 0.0f;
    cudaEventElapsedTime(&elapse, ev1, ev2);
    ////////////////////////////////////////////////////////////////////////////
    printf("time for gpu radixfind is %.3f ms\n",elapse);
    printf("my result:\n");
    for (int i = 0; i < topk; ++i){ 
        printf("%d, %10.10f\n", i, answer[i]);
    }

    cudaFree(p_d);
    printf("err: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;                
}

int main(){
    origin();
}