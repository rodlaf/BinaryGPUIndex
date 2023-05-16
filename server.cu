#include <cstdio>
#include <cstdlib>

#include "DeviceIndex.cu"
#include "crow.h"

#include <chrono>
#include <utility>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/lexical_cast.hpp>

using namespace boost::uuids;

int main() {
  const char *indexName = "test.index";
  int A10GCapacity = 950000000;
  int port = 80;

  crow::SimpleApp app;

  // Open index
  printf("Opening index...\n");
  DeviceIndex *index = new DeviceIndex(indexName, A10GCapacity);
  printf("Done.\n");

  CROW_ROUTE(app, "/insert")
      .methods("POST"_method)([](const crow::request &req) {
        auto x = crow::json::load(req.body);

        if (!x)
          return crow::response(crow::status::BAD_REQUEST);
        
        std::string idString = x["id"].s();
        std::string vectorString = x["vector"].s();

        uuid id = boost::lexical_cast<uuid>(idString);
        // convert string of 0s and 1s to uint64_cu vector
        uint64_cu vector = strtoull(vectorString.c_str(), NULL, 2);

        std::cout << id << std::endl;
        printBits(vector);

        return crow::response{crow::status::OK};
      });

  app.port(port).multithreaded().run();

  // Close index
  delete index;
}
