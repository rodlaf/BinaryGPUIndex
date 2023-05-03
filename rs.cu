#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <bitset>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>

typedef unsigned int uint32_cu;

__device__ uint32_cu hash(uint32_cu a) {
  a = (a ^ 61) ^ (a >> 16);
  a = a + (a << 3);
  a = a ^ (a >> 4);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15);
  return a;
}

__global__ void rand(int n, uint32_cu *xs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < n; i += stride)
    xs[idx] = hash(~idx);
}

__device__ uint32_cu positionBits(uint32_cu x, int position) {
  return (x >> ((sizeof(uint32_cu) - position) * 8)) & 0xff;
}

__global__ void collectHistogram(int n, uint32_cu *xs, int *histogram, int position) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < n; i += stride) {
    // expose relevant 8 bits as index to histogram
    uint32_cu bin = (xs[i] >> ((sizeof(uint32_cu) - position) * 8)) & 0xff;
    atomicAdd(&histogram[bin], 1);
  }
}

// /*
//   Copy values in xs that have bits in the specified position equal to those in 
//   the given pivot bin into the given temp memory.
// */
__global__ void relocate(int n, int position, uint32_cu bin, uint32_cu *xs, uint32_cu *temp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < n; i += stride) {
    if (positionBits(xs[i], position) == bin) {
      temp[idx] = xs[i];
    } else {
      temp[idx] = 0; // cudaMalloc does not zero out this memory for us
    }
  } 
}

void printBits(uint32_cu *x) {
  std::bitset<sizeof(uint32_cu) * CHAR_BIT> b(*x);
  std::cout << b << std::endl;
}

int main(){
  int n = 1 << 20;
  int k = 100000;

  int blockSize = 512;
  int numBlocks = (n + blockSize - 1) / blockSize;

  // generate random numbers
  uint32_cu *xs;
  cudaMallocManaged(&xs, n * sizeof(uint32_cu)); 

  rand<<<numBlocks, blockSize>>>(n, xs);
  cudaDeviceSynchronize();

  /////// Radix Select

  // alloc two temporary arrays
  uint32_cu *temp1, *temp2;
  cudaMallocManaged(&temp1, n * sizeof(uint32_cu));
  cudaMalloc(&temp2, n * sizeof(uint32_cu));

  // collect histogram
  int position = 1; // 4 positions total for a 32 bit unsigned integer
  int *histogram;
  cudaMallocManaged(&histogram, 256 * sizeof(int));

  collectHistogram<<<numBlocks, blockSize>>>(n, xs, histogram, position);
  cudaDeviceSynchronize();  

  // compute prefix sums
  int *prefixSums;
  cudaMallocManaged(&prefixSums, 256 * sizeof(int));

  thrust::inclusive_scan(thrust::device, histogram, histogram + 256, prefixSums);

  // find pivot bin
  int *pivotPtr = thrust::lower_bound(prefixSums, prefixSums + 256, k); 
  uint32_cu pivotBin = (uint32_cu)(pivotPtr - prefixSums);

  // relocate integers in that bin 
  relocate<<<numBlocks, blockSize>>>(n, position, pivotBin, xs, temp1);

  ///////


  for (int i = 0; i < 256; ++i){ 
      printf("%d: %d\n", i, prefixSums[i]);
  }

  printf("pivot: %d\n", (int)(pivotPtr - prefixSums));

  int count = 0;
  for (int i = 0; i < n; ++i) {
    if (temp1[i] != 0) {
      count++;
    }
  }

  printf("count: %d\n", count);

  printBits(&pivotBin);


  cudaFree(xs);
  cudaFree(histogram);
  cudaFree(prefixSums);
  cudaFree(temp1);
  cudaFree(temp2);

  return 0;    
}