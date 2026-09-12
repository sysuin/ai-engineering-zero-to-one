# Your first HTTP request. Four lines of work, and a great deal to notice.

import requests

from meridian_api import serve

BASE = serve()          # the book's own API, running on your machine

response = requests.get(f"{BASE}/revenue", params={"year": 2024, "quarter": 3}, timeout=10)

print("Status code:", response.status_code)
print("Content type:", response.headers["Content-Type"])
print()

data = response.json()          # JSON text -> a Python dictionary

print("Type of `data`:", type(data).__name__)
print("Keys:", list(data.keys()))
print("Total revenue:", data["total_revenue"])
print()

for region in data["regions"]:
    print(f"  {region['region']:11} {region['revenue']:>14,.2f}   {region['orders']:>5,} orders")
