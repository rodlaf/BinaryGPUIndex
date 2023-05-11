#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdio>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "kNearestNeighbors.h"

uint64_cu hash(uint64_cu h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53L;
  h ^= h >> 33;
  return h;
}

int main() {
  int numVectors = 1 << 20;

  std::ofstream f;
  f.open("test.txt", std::ios_base::app);

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  auto t1 = high_resolution_clock::now();

  // write to file 
  char *str = (char *)malloc(numVectors * (sizeof(boost::uuids::uuid) + sizeof(uint64_cu) + sizeof('\n')));
  for (int i = 0; i < numVectors; ++i) { 
    // generate uuid as key
    boost::uuids::uuid uuid = boost::uuids::random_generator()();

    // generate vector
    uint64_cu data;
    memcpy(&data, &uuid, 8);
    uint64_cu vector = hash(data);

    memcpy(str + i * 25, &uuid, 16);
    memcpy(str + i * 25 + 16, &vector, 8);
    memcpy(str + i * 25 + 24, "\n", 1);
  }
  f.write(str, 25 * numVectors);

  auto t2 = high_resolution_clock::now();
  auto ms_int = duration_cast<milliseconds>(t2 - t1);
  std::cout << ms_int.count() << "ms\n";

  f.close();

  std::ifstream rf("test.txt");

  t1 = high_resolution_clock::now();

  // read from file
  std::string rstr;
  std::string file_contents;
  while (std::getline(rf, rstr)) {
    file_contents += rstr;
    file_contents.push_back('\n');
  }

  t2 = high_resolution_clock::now();
  ms_int = duration_cast<milliseconds>(t2 - t1);
  std::cout << ms_int.count() << "ms\n";

  return 0;
}