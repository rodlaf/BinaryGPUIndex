#include <chrono>
#include <cstdio>

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
  using boost::uuids::random_generator;
  using boost::uuids::to_string;
  using boost::uuids::uuid;

  const char *vdbName = "test.txt";

  // open vector db
  VectorDB vdb(vdbName, 1 << 20);

  // generate random ids and vectors
  int numToAdd = 100;
  uuid ids[numToAdd];
  uint64_cu vectorsToAdd[numToAdd];
  for (int i = 0; i < numToAdd; ++i) {
    ids[i] = random_generator()();
    vectorsToAdd[i] = hash(~i);
  }

  // insert random ids and vectors
  vdb.insert(numToAdd, ids, vectorsToAdd);

  // query
  const int k = 10;
  uint64_cu queryVector = hash(~1);
  uint64_cu kNearestVectors[k];
  float kNearestDistances[k];
  uuid kNearestIds[k];

  vdb.query(queryVector, k, kNearestDistances, kNearestVectors, kNearestIds);

  // print results
  printf("Query: ");
  printBits(queryVector);
  for (int i = 0; i < k; ++i) {
    printf("%d: %s %8.8f ", i, to_string(kNearestIds[i]).c_str(),
           kNearestDistances[i]);
    printBits(kNearestVectors[i]);
  }

  // delete file
  std::remove(vdbName);

  return 0;
}