#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_map>

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

using boost::uuids::random_generator;
using boost::uuids::to_string;
using boost::uuids::uuid;

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
  // In practice, this means vector ids can't be too big. An alternate 
  // implementation could retrieve ids from disk instead. This would be much 
  // slower for large k when querying
  std::unordered_map<unsigned, uuid> idMap;

public:
  int numVectors;
  const char *name;

  // TODO: Make capacity public variable and ensure that it matches the passed
  // variable of the same name if a database is being reopened
  VectorDB(const char *nameParam, int capacity) {
    name = nameParam;
    
    // Allocate deviceKeys and initialize (initialization requires memory)
    cudaMalloc(&deviceKeys, capacity * sizeof(unsigned));
    thrust::sequence(thrust::device, deviceKeys, deviceKeys + capacity);

    // Allocate rest of on-device memory
    cudaMalloc(&workingMem1, capacity * sizeof(unsigned));
    cudaMalloc(&workingMem2, capacity * sizeof(unsigned));
    cudaMalloc(&workingMem3, capacity * sizeof(unsigned));
    cudaMalloc(&vectors, capacity * sizeof(uint64_cu));
    cudaMalloc(&deviceQueryVector, sizeof(uint64_cu));

    // Load vectors from db to device
    numVectors = 0;
    std::ifstream rf(name);
    std::string str;
    while (std::getline(rf, str)) {
      // Get id
      uuid id;
      memcpy(&id, str.c_str(), sizeof(uuid));

      // Get vector
      uint64_cu vector; 
      memcpy(&vector, str.c_str() + sizeof(uuid), sizeof(uint64_cu));

      printf("%s ", to_string(id).c_str());
      printBits(vector);

      // Record id in idMap
      idMap[numVectors] = id;
      numVectors++;
    }
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
    Inserts keys. Behaviour is undefined if ids already exist
  */
 // TODO: break into chunks
  void insert(int numToAdd, uuid ids[], uint64_cu vectorsToAdd[]) {
    // write ids and vectors to disk
    std::ofstream f;
    f.open(name, std::ios_base::app);
    int lineSize = sizeof(uuid) + sizeof(uint64_cu) + sizeof('\n');

    char *buffer = (char *)malloc(numToAdd * lineSize);

    for (int i = 0; i < numToAdd; ++i) {
      memcpy(buffer + i * lineSize, &ids[i], 16);
      memcpy(buffer + i * lineSize + sizeof(uuid), &vectorsToAdd[i], 8);
      memcpy(buffer + i * lineSize + sizeof(uuid) + sizeof(uint64_cu), "\n", 1);
    }
    f.write(buffer, numToAdd * lineSize);

    f.close();

    // insert ids into keymap
    for (int i = 0; i < numToAdd; ++i) {
      // TODO: Explain.
      idMap[numVectors + i] = ids[i];
    }

    // copy vectors to device
    cudaMemcpy(vectors + numVectors, vectorsToAdd, numToAdd * sizeof(uint64_cu),
               cudaMemcpyHostToDevice);

    // update numVectors
    numVectors += numToAdd;
  }

  /*

  */
  void query(uint64_cu &queryVector, int k, float kNearestDistances[],
             uint64_cu kNearestVectors[], uuid kNearestIds[]) {
    float *deviceKNearestDistances;
    unsigned *deviceKNearestKeys;
    uint64_cu *deviceKNearestVectors;
    cudaMalloc(&deviceKNearestDistances, k * sizeof(float));
    cudaMallocManaged(&deviceKNearestKeys, k * sizeof(unsigned));
    cudaMalloc(&deviceKNearestVectors, k * sizeof(uint64_cu));

    // copy query vector to device
    cudaMemcpy(deviceQueryVector, &queryVector, sizeof(uint64_cu),
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
      kNearestIds[i] = idMap[deviceKNearestKeys[i]];

    cudaFree(deviceKNearestDistances);
    cudaFree(deviceKNearestKeys);
    cudaFree(deviceKNearestVectors);
  }
};
