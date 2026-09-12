# What actually went over the wire. Everything else in this chapter is detail on this.

import requests

from meridian_api import serve

BASE = serve()

response = requests.get(f"{BASE}/revenue",
                        params={"year": 2024, "quarter": 3},
                        headers={"Accept": "application/json",
                                 "User-Agent": "meridian-book/1.0"},
                        timeout=10)

request = response.request
path = request.url.split("/", 3)[-1]

print("--- THE REQUEST ---")
print(f"{request.method} /{path} HTTP/1.1")
for name, value in request.headers.items():
    print(f"{name}: {value}")

print()
print("--- THE RESPONSE ---")
print(f"HTTP/1.1 {response.status_code} {response.reason}")
for name, value in response.headers.items():
    print(f"{name}: {value}")
print()
print(response.text[:180] + " ...")
