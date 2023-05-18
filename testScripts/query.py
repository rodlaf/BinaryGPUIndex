import random
import time
import requests

serverURL: str = "http://ec2-54-221-48-223.compute-1.amazonaws.com"
numRequests: int = 1000
topKPerRequest: int = 1000

def generateQueryBody(topK: int) -> dict:
    bodyDict = {}

    bodyDict["topK"] = topK
    bodyDict["vector"] = bin(random.randint(0, 2**64 - 1))[2:]

    return bodyDict

# import json 
# print(json.dumps(generateQueryBody(topK=topKPerRequest), indent=4))

print(f"Making {numRequests} queries with topK={topKPerRequest}...")

total: int = 0
for _ in range(numRequests):
    requestBody = generateQueryBody(topK=topKPerRequest)

    start = time.time()
    r: requests.Response = requests.post(
        serverURL + "/query", json=requestBody
    )
    end = time.time()
    total += end - start
    
    assert(r.status_code == 200)

print("Total time: {:.0f} ms.".format(1000 * total))
print("Per query average: {:.0f} ms.".format(1000 * total / numRequests))

