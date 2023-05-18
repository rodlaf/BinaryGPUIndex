#include <bitset>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <array>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <thrust/sequence.h>

#include "kNearestNeighbors.h"

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

class DeviceIndex {
private:
  // Pointers to device memory
  unsigned *workingMem1;
  unsigned *workingMem2;
  unsigned *workingMem3;
  uint64_cu *vectors;
  uint64_cu *deviceQueryVector;
  unsigned *deviceKeys; // sequential keys e.g., a range.

  // Use an in-memory hash map to keep track of deviceKey to vectorKey mappings
  // In practice, this means vector ids can't be too big. An alternate
  // implementation could retrieve ids from disk instead. This would be much
  // slower for large k when querying
  std::unordered_map<unsigned, uuid> idMap;

public:
  int numVectors = 0;
  const char *name;

  // Capacity must be passed as a maximum vector count as this enables
  // insertion and querying without allocation of memory every time.
  DeviceIndex(const char *nameParam, int capacity) {
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

    // Read vectors from file to device and idMap using a buffer
    int bufferSize = 4 << 20;
    uint64_cu *buffer = (uint64_cu *)malloc(bufferSize * sizeof(uint64_cu));
    int bufferCount = 0;
    auto flushBuffer = [&]() {
      cudaMemcpy(vectors + numVectors, buffer, bufferCount * sizeof(uint64_cu),
                 cudaMemcpyHostToDevice);
      numVectors += bufferCount;
      bufferCount = 0;
    };

    std::ifstream f(name);
    int lineSize = sizeof(uuid) + sizeof(uint64_cu);
    assert(lineSize == 24);
    char *lineBuf = (char *)malloc(lineSize);

    int lineCount = 0;
    while (f.read(lineBuf, lineSize)) {
      lineCount++;
      // TODO: implement upsert and not just insert here. Have defined behavior
      // if id already exists

      // Get id and record in idMap
      uuid id;
      memcpy(&id, lineBuf, sizeof(uuid));
      idMap[numVectors + bufferCount] = id;

      // Copy vector to buffer
      memcpy(buffer + bufferCount, lineBuf + sizeof(uuid), sizeof(uint64_cu));
      bufferCount++;

      // Flush buffer to device if full
      if (bufferCount == bufferSize)
        flushBuffer();
    }
    flushBuffer();

    free(buffer);
  }

  ~DeviceIndex() {
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
  void insert(int numToAdd, uuid ids[], uint64_cu vectorsToAdd[]) {
    // write ids and vectors to disk
    std::ofstream f;
    f.open(name, std::ios_base::app);
    int lineSize = sizeof(uuid) + sizeof(uint64_cu);

    char *buffer = (char *)malloc(numToAdd * lineSize);
    for (int i = 0; i < numToAdd; ++i) {
      memcpy(buffer + i * lineSize, &ids[i], sizeof(uuid));
      memcpy(buffer + i * lineSize + sizeof(uuid), &vectorsToAdd[i],
             sizeof(uint64_cu));
    }
    f.write(buffer, numToAdd * lineSize);
    f.close();
    free(buffer);

    // insert ids into keymap
    for (int i = 0; i < numToAdd; ++i) {
      // Store id in memory
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
