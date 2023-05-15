#include <cstdio>

#include "DeviceIndex.cu"
#include "crow.h"

int main() {
  crow::SimpleApp app;

  const char *indexName = "test.index";
  int A10GCapacity = 950000000;
  int port = 80;

  DeviceIndex *index = new DeviceIndex(indexName, A10GCapacity);

  CROW_ROUTE(app, "/insert")
      .methods("PUT"_method)([](const crow::request &req) {
        auto x = crow::json::load(req.body);
        std::cout << x;
        if (!x)
          return crow::response(crow::status::BAD_REQUEST);
        return crow::response{crow::status::OK};
      });

  app.port(port).multithreaded().run();

  delete index;
}
