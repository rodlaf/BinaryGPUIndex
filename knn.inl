#include <bitset>
#include <iostream>
#include <math.h>
#include <sys/time.h>

#include "radix_select.h"

typedef unsigned long long int uint64_cu;

__device__ void cosineDistance(uint64_cu *a, uint64_cu *b, unsigned *dest) {
  // __popcll computes the Hamming Weight of an integer (e.g., number of bits
  // that are 1)
  float a_dot_b = (float)__popcll(*a & *b);
  float a_dot_a = (float)__popcll(*a);
  float b_dot_b = (float)__popcll(*b);

  *dest = (unsigned)((1 - (a_dot_b / (sqrt(a_dot_a) * sqrt(b_dot_b)))) * UINT32_MAX);
}

__global__ void computeDistances(int numIndexes, uint64_cu *query,
                                 uint64_cu *indexes, unsigned *distances) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numIndexes; i += stride)
    cosineDistance(query, &indexes[i], &distances[i]);
}

void kNearestNeighbors(uint64_cu *vectors, unsigned *keys, uint64_cu *query,
                       int numVectors, int k, float *kNearestDistances,
                       uint64_cu *kNearestVectors, unsigned *kNearestKeys,
                       unsigned *distances, unsigned *workingMem1,
                       unsigned *workingMem2) {
  int blockSize = 1024;
  int numBlocks = (numVectors + blockSize - 1) / blockSize;

  unsigned *uintKNearestDistances;
  uintKNearestDistances = (unsigned *)malloc(k * sizeof(unsigned));

  // compute distances
  computeDistances<<<numBlocks, blockSize>>>(numVectors, query, vectors,
                                             distances);
  cudaDeviceSynchronize();

  // select smallest `k` distances
  radixSelect(distances, keys, numVectors, k, uintKNearestDistances, kNearestKeys,
               workingMem1, workingMem2);

  for (int i = 0; i < k; ++i) {
    // convert unsigned integer distances to floating point distances
    kNearestDistances[i] = (float)uintKNearestDistances[i] / (float)UINT_MAX;

    // copy indicated indexes from device to host
    int idx = kNearestKeys[i];
    cudaMemcpy(&kNearestVectors[i], &vectors[idx], sizeof(uint64_cu),
               cudaMemcpyDeviceToHost);
  }

  free(uintKNearestDistances);
}