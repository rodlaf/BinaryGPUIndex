#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <utility>

#include <boost/lexical_cast.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "DeviceIndex.cu"
#include "crow.h"

using namespace boost::uuids;

std::string vectorToString(uint64_cu vector) {
  return std::bitset<64>(vector).to_string();
}

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

  // Insert route
  CROW_ROUTE(app, "/insert")
      .methods("POST"_method)([&](const crow::request &req) {
        auto jsonBody = crow::json::load(req.body);

        if (!jsonBody["vectors"])
          return crow::response(crow::status::BAD_REQUEST);

        int numToInsert = jsonBody["vectors"].size();

        uuid *ids = (uuid *)malloc(numToInsert * sizeof(uuid));
        uint64_cu *vectors =
            (uint64_cu *)malloc(numToInsert * sizeof(uint64_cu));

        // Retrieve ids and vectors
        for (int i = 0; i < numToInsert; ++i) {
          std::string idString = jsonBody["vectors"][i]["id"].s();
          std::string vectorString = jsonBody["vectors"][i]["values"].s();

          uuid id = boost::lexical_cast<uuid>(idString);
          uint64_cu vector = strtoull(vectorString.c_str(), NULL, 2);

          ids[i] = id;
          vectors[i] = vector;
        }

        // insert ids and vectors into index
        index->insert(numToInsert, ids, vectors);

        free(ids);
        free(vectors);

        crow::json::wvalue response(
            {{"insertedCount", std::to_string(numToInsert)}});

        return crow::response{response};
      });

  // Query route
  CROW_ROUTE(app, "/query")
      .methods("POST"_method)([&](const crow::request &req) {
        auto jsonBody = crow::json::load(req.body);

        if (!jsonBody["topK"] || !jsonBody["vector"])
          return crow::response(crow::status::BAD_REQUEST);

        // Retrieve topK and vector
        int topK = jsonBody["topK"].i();
        std::string vectorString = jsonBody["vector"].s();
        uint64_cu vector = strtoull(vectorString.c_str(), NULL, 2);

        // Make sure topK is within bounds
        if (topK > index->numVectors)
          return crow::response(crow::status::BAD_REQUEST);

        // Set up memory to collect query results
        uint64_cu *kNearestVectors =
            (uint64_cu *)malloc(topK * sizeof(uint64_cu));
        float *kNearestDistances = (float *)malloc(topK * sizeof(float));
        uuid *kNearestIds = (uuid *)malloc(topK * sizeof(uuid));

        // Query index
        index->query(vector, topK, kNearestDistances, kNearestVectors,
                     kNearestIds);

        crow::json::wvalue response({});

        for (int i = 0; i < topK; ++i) {
          response["matches"][i]["id"] = to_string(kNearestIds[i]);
          response["matches"][i]["distance"] =
              std::to_string(kNearestDistances[i]);
          response["matches"][i]["values"] = vectorToString(kNearestVectors[i]);
        }

        free(kNearestVectors);
        free(kNearestDistances);
        free(kNearestIds);

        return crow::response{response};
      });

  // Run server
  printf("Server is running on port %d.\n", port);
  app.port(port).run();

  // Close index
  delete index;
}
