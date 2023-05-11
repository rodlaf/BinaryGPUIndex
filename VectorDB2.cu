#include <cstdio>
#include <unordered_map>
#include <string>

#include <thrust/sequence.h>

#include "kNearestNeighbors.h"

// Design decision: Separate deviceKeys and vectorKeys.
//
// vectorKeys can be very long (e.g., a whole UUID) but deviceKeys must
// be a single unsigned integer (e.g., 32 bits). This imposes a limitation
// that at most 2^32 billion vectors can be searched on a GPU at a time, no
// matter the memory constraints of the GPU. This can be changed in the future.
//
// a separate collection of deviceKey-to-vectorkey key-value pairs will be kept
// in either CPU memory or the persistent key value store (e.g., RocksDB).
//
// in addition, deviceKeys will be sequential in order to enable quick vector
// retrieval by interpreting them as indexes in the on-device vector array

// Requires keys to be sequential, representing array indexes
__global__ void retrieveVectorsFromKeys(uint64_cu *vectors, unsigned *keys,
                                        int numKeys, uint64_cu *retrieved) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < numKeys; i += stride)
    retrieved[i] = vectors[keys[i]];
}

void printBits(uint64_cu &x) {
  std::bitset<sizeof(uint64_cu) * CHAR_BIT> b(x);
  std::cout << b << std::endl;
}

class VectorDB {
private:
  // Pointers to device memory
  unsigned *workingMem1;
  unsigned *workingMem2;
  unsigned *workingMem3;
  uint64_cu *vectors;
  uint64_cu *deviceQueryVector;
  unsigned *deviceKeys; // sequential keys

  // Use an in-memory hash map to keep track of deviceKey to vectorKey mappings
  std::unordered_map<unsigned, std::string> keyMap;

public:
  int numVectors;

  // TODO: Make capacity public variable and ensure that it matches the passed
  // variable of the same name if a database is being reopened
  VectorDB(const std::string &name, int capacity) {
    // CUDA INITIALIZATION
    // Allocate all on-device memory
    cudaMalloc(&workingMem1, capacity * sizeof(unsigned));
    cudaMalloc(&workingMem2, capacity * sizeof(unsigned));
    cudaMalloc(&workingMem3, capacity * sizeof(unsigned));
    cudaMallocManaged(&vectors, capacity * sizeof(uint64_cu));
    cudaMallocManaged(&deviceQueryVector, sizeof(uint64_cu));
    cudaMalloc(&deviceKeys, capacity * sizeof(unsigned));

    // Initialize device keys
    thrust::sequence(thrust::device, deviceKeys, deviceKeys + capacity);

    // Load vectors from db to device

  }

  ~VectorDB() {
    cudaFree(workingMem1);
    cudaFree(workingMem2);
    cudaFree(workingMem3);
    cudaFree(vectors);
    cudaFree(deviceQueryVector);
    cudaFree(deviceKeys);
  }

  /*
    Inserts new key. Panics if key already exists
  */
  void insert(int numVectors, const char **vectorKeys, uint64_cu *vectors) {
    // NOTE: These two should eventually be made into a transaction
    // Check if the vectorKey exists
    for (int i = 0; i < numVectors; ++i) {
      printf("vectorKey: %s, vector: ", vectorKeys[i]);
      printBits(vectors[i]);
    }
  }

  /*
  
  */
  void query(uint64_cu *queryVector, int k, float *kNearestDistances,
             uint64_cu *kNearestVectors, std::string kNearestVectorKeys[]) {
    float *deviceKNearestDistances;
    unsigned *deviceKNearestKeys;
    uint64_cu *deviceKNearestVectors;
    cudaMallocManaged(&deviceKNearestDistances, k * sizeof(float));
    cudaMallocManaged(&deviceKNearestKeys, k * sizeof(unsigned));
    cudaMalloc(&deviceKNearestVectors, k * sizeof(uint64_cu));

    // copy query vector to device
    cudaMemcpy(deviceQueryVector, queryVector, sizeof(uint64_cu),
               cudaMemcpyHostToDevice);

    kNearestNeighbors(vectors, deviceKeys, deviceQueryVector, numVectors, k,
                      deviceKNearestDistances, deviceKNearestKeys, workingMem1,
                      workingMem2, workingMem3);

    // retrieve vectors from relevant keys
    retrieveVectorsFromKeys<<<1, 1024>>>(vectors, deviceKNearestKeys, k,
                                         deviceKNearestVectors);
    cudaDeviceSynchronize();

    // copy solution from device to host specified by caller
    cudaMemcpy(kNearestDistances, deviceKNearestDistances, k * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(kNearestVectors, deviceKNearestVectors, k * sizeof(uint64_cu),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < k; ++i)
      kNearestVectorKeys[i] = keyMap[deviceKNearestKeys[i]];

    cudaFree(deviceKNearestDistances);
    cudaFree(deviceKNearestKeys);
    cudaFree(deviceKNearestVectors);
  }
};
