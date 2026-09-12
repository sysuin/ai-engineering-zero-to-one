# A request with no timeout can hang forever. This is the single most common way a
# scheduled job silently stops working.

import requests

from meridian_api import serve

BASE = serve()

try:
    requests.get(f"{BASE}/slow", params={"seconds": 5}, timeout=1.5)
except requests.exceptions.Timeout:
    print("Timed out after 1.5s, as instructed. The program is still in control.")

# Give it enough time and the same call succeeds.
response = requests.get(f"{BASE}/slow", params={"seconds": 0.5}, timeout=5)
print("With a longer timeout:", response.status_code, response.json())

print()
print("Rule: every request in this book passes `timeout=`. There is no exception.")
