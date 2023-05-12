#include <chrono>
#include <cstdio>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "VectorDB.cu"

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
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  using boost::uuids::random_generator;
  using boost::uuids::to_string;
  using boost::uuids::uuid;

  const char *vdbName = "test.txt";
  int vdbCapacity = 5 << 20;

  // open vector db
  printf("Opening...\n");
  auto t1 = high_resolution_clock::now();
  VectorDB *vdb = new VectorDB(vdbName, vdbCapacity);
  auto t2 = high_resolution_clock::now();
  auto ms_int = duration_cast<milliseconds>(t2 - t1);
  printf("Done. Execution time: %ldms.\n", ms_int.count());

  // TODO: Do generation-insertion in batches, not just insertion

  // generate random ids and vectors
  int numToAdd = 3500000;
  // use heap since these arrays are huge
  printf("Generating..\n");
  t1 = high_resolution_clock::now();
  uuid *ids = (uuid *)malloc(numToAdd * sizeof(uuid));
  uint64_cu *vectorsToAdd = (uint64_cu *)malloc(numToAdd * sizeof(uint64_cu));
  for (int i = 0; i < numToAdd; ++i) {
    ids[i] = random_generator()();
    vectorsToAdd[i] = hash(~i);
  }
  t2 = high_resolution_clock::now();
  ms_int = duration_cast<milliseconds>(t2 - t1);
  printf("Done. Execution time: %ldms.\n", ms_int.count());

  // insert random ids and vectors
  printf("Inserting...\n");
  t1 = high_resolution_clock::now();
  int chunkSize = 4 << 20;
  int numChunks = (numToAdd + chunkSize - 1) / chunkSize;
  printf("numChunks: %d\n", numChunks);
  for (int i = 0; i < numChunks; ++i) {
    int start = i * chunkSize;
    int numInChunk = chunkSize;
    if (i == numChunks - 1) {
      numInChunk = numToAdd % chunkSize;
    }
    // printf("numInChunk: %d\n", numInChunk);
    vdb->insert(numInChunk, ids + start, vectorsToAdd + start);
  }
  t2 = high_resolution_clock::now();
  ms_int = duration_cast<milliseconds>(t2 - t1);
  printf("Done. Execution time: %ldms.\n", ms_int.count());

  free(ids);
  free(vectorsToAdd);

  // query
  const int k = 10;
  uint64_cu queryVector = hash(~1);
  uint64_cu kNearestVectors[k];
  float kNearestDistances[k];
  uuid kNearestIds[k];

  printf("Querying...\n");
  t1 = high_resolution_clock::now();
  vdb->query(queryVector, k, kNearestDistances, kNearestVectors, kNearestIds);
  t2 = high_resolution_clock::now();
  ms_int = duration_cast<milliseconds>(t2 - t1);
  printf("Done. Execution time: %ldms.\n", ms_int.count());

  // print results
  printf("Query: ");
  printBits(queryVector);
  for (int i = 0; i < k; ++i) {
    printf("%d: %s %8.8f ", i, to_string(kNearestIds[i]).c_str(),
           kNearestDistances[i]);
    printBits(kNearestVectors[i]);
  }

  // close db
  delete vdb;

  // reopen
  printf("Reopening...\n");
  t1 = high_resolution_clock::now();
  VectorDB *vdb2 = new VectorDB(vdbName, vdbCapacity);
  t2 = high_resolution_clock::now();
  ms_int = duration_cast<milliseconds>(t2 - t1);
  printf("Done. Execution time: %ldms.\n", ms_int.count());

  // query again
  printf("Querying again...\n");
  t1 = high_resolution_clock::now();
  vdb2->query(queryVector, k, kNearestDistances, kNearestVectors, kNearestIds);
  t2 = high_resolution_clock::now();
  ms_int = duration_cast<milliseconds>(t2 - t1);
  printf("Done. Execution time: %ldms.\n", ms_int.count());

  // print results
  printf("Query: ");
  printBits(queryVector);
  for (int i = 0; i < k; ++i) {
    printf("%d: %s %8.8f ", i, to_string(kNearestIds[i]).c_str(),
           kNearestDistances[i]);
    printBits(kNearestVectors[i]);
  }

  // close second db
  delete vdb2;

  // delete file
  std::remove(vdbName);

  return 0;
}