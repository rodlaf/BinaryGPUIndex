#include <cstdio>
#include <chrono>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "VectorDB2.cu"

// murmur64 hash function
uint64_cu hash(uint64_cu h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53L;
  h ^= h >> 33;
  return h;
}

int main(void) {
  using boost::uuids::uuid;
  using boost::uuids::random_generator;

  // open vector db
  VectorDB vdb("test.txt", 1000);

  // generate random ids and vectors
  int numVectors = 100;
  uuid *ids = (uuid *)malloc(numVectors * sizeof(uuid));
  uint64_cu *vectors = (uint64_cu *)malloc(numVectors * sizeof(uint64_cu));
  for (int i = 0; i < numVectors; ++i) {
    ids[i] = random_generator()();
    vectors[i] = hash(~i);
  }

  // insert random ids and vectors
  vdb.insert(numVectors, ids, vectors);

  // query
  const int k = 10;
  uint64_cu queryVector = hash(~1);
  uint64_cu *kNearestVectors = (uint64_cu *)malloc(k * sizeof(uint64_cu));
  float *kNearestDistances = (float *)malloc(k * sizeof(float));
  uuid kNearestIds[k];

  vdb.query(&queryVector, k, kNearestDistances, kNearestVectors, kNearestIds);

  // print results
  printf("Query: ");
  printBits(queryVector);
  for (int i = 0; i < k; ++i) {
    printf("%d: %s %8.8f ", i, boost::uuids::to_string(kNearestIds[i]).c_str(), kNearestDistances[i]);
    printBits(kNearestVectors[i]);
  }

  free(kNearestVectors);

  return 0;
}