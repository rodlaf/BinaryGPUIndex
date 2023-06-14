#include <bitset>
#include <iostream>
#include <math.h>
#include <sys/time.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include "radixSelect.h"

using namespace thrust::placeholders;

/*
  Multiplies a consecutive array of vectors by a single vector. All vectors
  must have size D. This method is used as an alternative to complicated
  usage of thrust::transform with strided iterators, etc.

  V is an array of N vectors represented as N * D floats (T type)
  Q is a single vectors represented as D floats (T type)
*/
template <typename T>
__global__ void multiplyManyBySingle(T *V, T *Q, size_t N, size_t D,
                                     T *results) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
    results[i] = V[i] * Q[i % D];
}

/*
  Conversions to and from floats to unsigned integers for ranking with
  radix select.
*/
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

// unary operator used in thrust::transform in norm function
struct squareRoot : std::unary_function<float, float> {
  __device__ bool operator()(float x) const { return pow(x, 0.5); }
};

/*
  Computes norms of sequentially stored vectors of equal length.

  wMem must have size N * D * sizeof(T)
  results must have size N * sizeof(float)
*/
template <typename T>
void norms(T *V, size_t N, size_t D, T *wMem, float *results) {
  // square elements
  thrust::transform(thrust::device, V, V + N * D, wMem,
                    thrust::square<float>());
  // sum elements
  thrust::reduce_by_key(thrust::device,
                        thrust::make_transform_iterator(
                            thrust::counting_iterator<int>(0), _1 / D),
                        thrust::make_transform_iterator(
                            thrust::counting_iterator<int>(N * D), _1 / D),
                        V, thrust::discard_iterator<int>(), results);
  // square root results
  thrust::transform(thrust::device, results, results + N, results,
                    squareRoot());
}

// unary operator functor used in thrust::transform
struct divisionFunctor {
  float divisor;

  divisionFunctor(float _divisor) : divisor(_divisor){};

  __device__ float operator()(float &x) const { return x / divisor; }
};

/*
  Compute cosine distances.

  wMem must have size N * D * sizeof(T)
  distances must have size N * sizeof(float)
*/
template <typename T>
void cosineDistances(T *vectors, size_t D, T *query, int N, T *wMem,
                     float *distances) {
  int numElts = N * D;
  int blockSize = 1024;
  int numBlocks = (numElts + blockSize - 1) / blockSize;

  float *vectorNorms;
  cudaMalloc(&vectorNorms, N * sizeof(float));

  float *queryNorm;
  cudaMallocManaged(&queryNorm, sizeof(float));

  // compute vector norms
  // IMPORTANT NOTE: This computation can be cached, and should be. Vector
  // norms need not be computed every time a query is made, only once after
  // the vector(s) are inserted.
  norms(vectors, N, D, wMem, vectorNorms);

  // compute query norm
  norms(query, 1, D, wMem, queryNorm);

  // compute inner product of vectors by the query into distances
  multiplyManyBySingle<T><<<numBlocks, blockSize>>>(vectors, query, N, D, wMem);
  thrust::reduce_by_key(thrust::device,
                        thrust::make_transform_iterator(
                            thrust::counting_iterator<int>(0), _1 / D),
                        thrust::make_transform_iterator(
                            thrust::counting_iterator<int>(N * D), _1 / D),
                        wMem, thrust::discard_iterator<int>(), distances);

  // divide results by vector norms
  thrust::transform(thrust::device, distances, distances + N, vectorNorms,
                    distances, thrust::divides<float>());

  // divide results by query norm
  thrust::transform(thrust::device, distances, distances + N, distances,
                    divisionFunctor(*queryNorm));

  cudaFree(vectorNorms);
  cudaFree(queryNorm);
}

/*
  Select K Nearest Neighbors of a query vector from a set of vectors of equal
  length.

  NOTE: Only full precision floats currently supported. This may be extended
  but will involve working closely with the capabilities of the thrust library
  and writing custom code for its shortcomings regarding half-precision floats.
*/
template <typename T>
void kNearestNeighbors(T *vectors, size_t D, unsigned *keys, T *query, int N,
                       int K, float *kNearestDistances, unsigned *kNearestKeys,
                       unsigned *workingMem1, unsigned *workingMem2,
                       unsigned *workingMem3) {
  int numElts = N * D;
  int blockSize = 1024;
  int numBlocks = (numElts + blockSize - 1) / blockSize;

  // use working memory to compute distances
  float *distances = (float *)workingMem1;
  unsigned *uintDistances = workingMem2;

  // collect the best distances in their unsigned integer versions
  unsigned *uintKNearestDistances;
  cudaMalloc(&uintKNearestDistances, K * sizeof(unsigned));

  // allocate huge chunk of working memory on device
  // NOTE: this can and should be passed as pre-allocated memory such that it
  // is not allocated every time it is needed.
  T *wMem;
  cudaMalloc(&wMem, N * D * sizeof(T));

  // compute distances
  cosineDistances(vectors, D, query, N, wMem, distances);

  // convert distances to unsigned integers
  floatToUnsigned<<<numBlocks, blockSize>>>(distances, uintDistances, N);
  cudaDeviceSynchronize();

  // select smallest K distances
  radixSelect(uintDistances, keys, N, K, uintKNearestDistances, kNearestKeys,
              workingMem1, workingMem3);

  // convert unsigned integer distances back to floating point distances
  unsignedToFloat<<<1, blockSize>>>(uintKNearestDistances, kNearestDistances,
                                    K);
  cudaDeviceSynchronize();

  cudaFree(uintKNearestDistances);
  cudaFree(wMem);
}
