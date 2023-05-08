#include <bitset>
#include <iostream>
#include <math.h>
#include <sys/time.h>

#include "radixSelect.h"

typedef unsigned long long int uint64_cu;

__device__ void cosineDistance(uint64_cu *a, uint64_cu *b, unsigned *dest) {
  // __popcll computes the Hamming Weight of an integer (e.g., number of bits
  // that are 1)
  float a_dot_b = (float)__popcll(*a & *b);
  float a_dot_a = (float)__popcll(*a);
  float b_dot_b = (float)__popcll(*b);

  // returns cosine distance as an unsigned integer. this will get turned into
  // a float once the k smallest are selected further on.
  *dest = (unsigned)((1 - (a_dot_b / (sqrt(a_dot_a) * sqrt(b_dot_b)))) *
                     UINT32_MAX);
}

__global__ void computeDistances(int numIndexes, uint64_cu *query,
                                 uint64_cu *indexes, unsigned *distances) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numIndexes; i += stride)
    cosineDistance(query, &indexes[i], &distances[i]);
}

__global__ void unsignedToFloat(unsigned *uintValues, float *fValues,
                                int numValues) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numValues; i += stride)
    fValues[i] = (float)uintValues[i] / (float)UINT_MAX;
}

__global__ void retrieveVectorsFromKeys(uint64_cu *vectors, unsigned *keys,
                                        int k, uint64_cu *retrieved) {

}

void kNearestNeighbors(uint64_cu *vectors, unsigned *keys, uint64_cu *query,
                       int numVectors, int k, float *kNearestDistances,
                       uint64_cu *kNearestVectors, unsigned *kNearestKeys,
                       unsigned *workingMem1, unsigned *workingMem2,
                       unsigned *workingMem3) {
  int blockSize = 1024;
  int numBlocks = (numVectors + blockSize - 1) / blockSize;

  // we first collect the best distances in their unsigned integer versions
  unsigned *uintKNearestDistances;
  cudaMalloc(&uintKNearestDistances, k * sizeof(unsigned));

  // use working memory to compute distances
  unsigned *distances = workingMem3;

  // compute distances
  computeDistances<<<numBlocks, blockSize>>>(numVectors, query, vectors,
                                             distances);
  cudaDeviceSynchronize();

  // select smallest `k` distances
  radixSelect(distances, keys, numVectors, k, uintKNearestDistances,
              kNearestKeys, workingMem1, workingMem2);

  // convert unsigned integer distances to floating point distances
  unsignedToFloat<<<1, blockSize>>>(uintKNearestDistances, kNearestDistances,
                                    k);
  cudaDeviceSynchronize();

  // for (int i = 0; i < k; ++i) {
  //   // convert unsigned integer distances to floating point distances
  //   kNearestDistances[i] = (float)uintKNearestDistances[i] / (float)UINT_MAX;

  //   // copy indicated indexes from device to host
  //   int idx = kNearestKeys[i];
  //   cudaMemcpy(&kNearestVectors[i], &vectors[idx], sizeof(uint64_cu),
  //              cudaMemcpyDeviceToHost);
  // }

  cudaFree(uintKNearestDistances);
}