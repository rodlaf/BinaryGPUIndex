#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include <boost/lexical_cast.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "DeviceIndex.cu"
#include "crow.h"

using namespace boost::uuids;

int main() {
  const char *indexName = "test.index";
  int A10GCapacity = 950000000;
  int port = 80;

  crow::SimpleApp app;
  app.loglevel(crow::LogLevel::Warning);

  // Open index
  printf("Opening index...\n");
  DeviceIndex *index = new DeviceIndex(indexName, A10GCapacity);
  printf("Done.\n");

  CROW_ROUTE(app, "/insert")
      .methods("POST"_method)([&](const crow::request &req) {
        auto x = crow::json::load(req.body);

        if (!x["vectors"])
          return crow::response(crow::status::BAD_REQUEST);

        int numToInsert = x["vectors"].size();

        uuid *ids = (uuid *)malloc(numToInsert * sizeof(uuid));
        uint64_cu *vectors =
            (uint64_cu *)malloc(numToInsert * sizeof(uint64_cu));

        // Retrieve ids and vectors
        for (int i = 0; i < numToInsert; ++i) {
          std::string idString = x["vectors"][i]["id"].s();
          std::string vectorString = x["vectors"][i]["values"].s();

          uuid id = boost::lexical_cast<uuid>(idString);
          uint64_cu vector = strtoull(vectorString.c_str(), NULL, 2);

          ids[i] = id;
          vectors[i] = vector;
        }

        free(ids);
        free(vectors);

        // insert ids and vectors into index
        index->insert(numToInsert, ids, vectors);

        crow::json::wvalue response(
            {{"insertedCount", std::to_string(numToInsert)}});

        return crow::response{response};
      });

  printf("Server is running on port %d.\n", port);
  app.port(port).multithreaded().run();

  // Close index
  delete index;
}
