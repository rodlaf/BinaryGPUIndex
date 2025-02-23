#include <chrono>
#include <cstdio>

#include "DeviceIndex.cu"

// murmur64 hash function
uint64_cu hash(uint64_cu h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53L;
  h ^= h >> 33;
  return h;
}

/*
  This test opens an index, inserts vectors into it, makes a query, and closes it.
  It then re-opens the same index, and makes the same query as before, printing
  the output to insure that the results are the same.

  Different steps in this test can be commented out to accomplish different
  things, such as generating a random index
*/

int main(void) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  using boost::uuids::random_generator;
  using boost::uuids::to_string;
  using boost::uuids::uuid;

  const char *vdbName = "big.index";
  int vdbCapacity = 950000000;
  int numToInsert = 500000000;

  // open vector db
  printf("Opening...\n");
  auto t1 = high_resolution_clock::now();
  DeviceIndex *vdb = new DeviceIndex(vdbName, vdbCapacity);
  auto t2 = high_resolution_clock::now();
  auto ms_int = duration_cast<milliseconds>(t2 - t1);
  printf("Done. Execution time: %ldms.\n", ms_int.count());

  int batchSize = 4 << 20;
  int numBatches = (numToInsert + batchSize - 1) / batchSize;

  // use heap since these arrays are huge
  uuid *ids = (uuid *)malloc(batchSize * sizeof(uuid));
  uint64_cu *vectorsToAdd = (uint64_cu *)malloc(batchSize * sizeof(uint64_cu));

  for (int batch = 0; batch < numBatches; ++batch) {
    // Adjust batch size if last batch
    if (batch == numBatches - 1)
      batchSize = numToInsert % batchSize;

    printf("Generating and inserting (%d\\%d)...\n", batch, numBatches);
    t1 = high_resolution_clock::now();

    for (int i = 0; i < batchSize; ++i) {
      ids[i] = random_generator()();
      vectorsToAdd[i] = hash((batch + 1) * (i + 1));
    }

    vdb->insert(batchSize, ids, vectorsToAdd);

    t2 = high_resolution_clock::now();
    ms_int = duration_cast<milliseconds>(t2 - t1);
    printf("Done. Execution time: %ldms.\n", ms_int.count());
  }

  free(ids);
  free(vectorsToAdd);

  // query
  const int k = 2;
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

  // // print results
  // printf("Query: ");
  // printBits(queryVector);
  // for (int i = 0; i < k; ++i) {
  //   printf("%d: %8.8f %s ", i, kNearestDistances[i],
  //          to_string(kNearestIds[i]).c_str());
  //   printBits(kNearestVectors[i]);
  // }

  // // close db
  // delete vdb;

  // // reopen
  // printf("Reopening...\n");
  // t1 = high_resolution_clock::now();
  // DeviceIndex *vdb2 = new DeviceIndex(vdbName, vdbCapacity);
  // t2 = high_resolution_clock::now();
  // ms_int = duration_cast<milliseconds>(t2 - t1);
  // printf("Done. Execution time: %ldms.\n", ms_int.count());

  // // query again
  // printf("Querying again...\n");
  // t1 = high_resolution_clock::now();
  // vdb2->query(queryVector, k, kNearestDistances, kNearestVectors, kNearestIds);
  // t2 = high_resolution_clock::now();
  // ms_int = duration_cast<milliseconds>(t2 - t1);
  // printf("Done. Execution time: %ldms.\n", ms_int.count());

  // // print results
  // printf("Query: ");
  // printBits(queryVector);
  // for (int i = 0; i < k; ++i) {
  //   printf("%d: %8.8f %s ", i, kNearestDistances[i],
  //          to_string(kNearestIds[i]).c_str());
  //   printBits(kNearestVectors[i]);
  // }

  // // close second db
  // delete vdb2;

  // // delete file
  // std::remove(vdbName);

  return 0;
}