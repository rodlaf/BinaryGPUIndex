#include <bitset>
#include <iostream>
#include <math.h>
#include <sys/time.h>

#include "radixSelect.h"

typedef unsigned long long int uint64_cu;

/*
  We can shave off ~20ms of each call by performing the float to unsigned 
  conversion directly in this function. We don't do this to show that the
  algorithm is completely agnositc to the distance metric used, whether 
  it returns a floating point value or an integer.
*/
__device__ void cosineDistance(uint64_cu *a, uint64_cu *b, float *dest) {
  // __popcll computes the Hamming Weight of an integer (e.g., number of bits
  // that are 1)
  float a_dot_b = (float)__popcll(*a & *b);
  float a_dot_a = (float)__popcll(*a);
  float b_dot_b = (float)__popcll(*b);

  *dest = (1 - (a_dot_b / (sqrt(a_dot_a) * sqrt(b_dot_b))));
}

__global__ void computeDistances(int numIndexes, uint64_cu *query,
                                 uint64_cu *indexes, float *distances) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numIndexes; i += stride)
    cosineDistance(query, &indexes[i], &distances[i]);
}

__global__ void floatToUnsigned(float *fValues, unsigned *uintValues, 
                                int numValues) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numValues; i += stride)
    uintValues[i] = (unsigned)(fValues[i] * UINT_MAX);
}

__global__ void unsignedToFloat(unsigned *uintValues, float *fValues,
                                int numValues) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numValues; i += stride)
    fValues[i] = (float)uintValues[i] / (float)UINT_MAX;
}

__global__ void retrieveVectorsFromKeys(uint64_cu *vectors, unsigned *keys,
                                        int numKeys, uint64_cu *retrieved) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numKeys; i += stride)
    retrieved[i] = vectors[keys[i]];
}

void kNearestNeighbors(uint64_cu *vectors, unsigned *keys, uint64_cu *query,
                       int numVectors, int k, float *kNearestDistances,
                       uint64_cu *kNearestVectors, unsigned *kNearestKeys,
                       unsigned *workingMem1, unsigned *workingMem2,
                       unsigned *workingMem3) {
  int blockSize = 1024;
  int numBlocks = (numVectors + blockSize - 1) / blockSize;

  // use working memory to compute distances
  float *distances = (float *)workingMem1;
  unsigned *uintDistances = workingMem2;

  // collect the best distances in their unsigned integer versions
  unsigned *uintKNearestDistances;
  cudaMalloc(&uintKNearestDistances, k * sizeof(unsigned));

  // compute distances
  computeDistances<<<numBlocks, blockSize>>>(numVectors, query, vectors,
                                             distances);
  cudaDeviceSynchronize();

  // convert distances to unsigned integers
  floatToUnsigned<<<numBlocks, blockSize>>>(distances, uintDistances, numVectors);
  cudaDeviceSynchronize();

  // select smallest `k` distances
  radixSelect(uintDistances, keys, numVectors, k, uintKNearestDistances,
              kNearestKeys, workingMem1, workingMem3);

  // convert unsigned integer distances back to floating point distances
  unsignedToFloat<<<1, blockSize>>>(uintKNearestDistances, kNearestDistances,
                                    k);
  cudaDeviceSynchronize();

  // retrieve vectors from relevant keys
  retrieveVectorsFromKeys<<<1, blockSize>>>(vectors, kNearestKeys, k,
                                            kNearestVectors);
  cudaDeviceSynchronize();

  cudaFree(uintKNearestDistances);
}