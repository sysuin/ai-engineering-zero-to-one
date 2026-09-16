# Two ways to page through a list that is changing while you read it.

import requests

from meridian_api import serve

BASE = serve()
LIMIT = 10


def publish(count):
    requests.post(f"{BASE}/feed", json={"count": count}, timeout=5)


# --- Offset pagination: "skip N, give me the next ten".
seen = []
page1 = requests.get(f"{BASE}/feed", params={"offset": 0, "limit": LIMIT}, timeout=5).json()
seen += [item["id"] for item in page1["items"]]
publish(3)                                    # three new tickets arrive between pages
page2 = requests.get(f"{BASE}/feed", params={"offset": LIMIT, "limit": LIMIT},
                     timeout=5).json()
seen += [item["id"] for item in page2["items"]]

duplicates = sorted({i for i in seen if seen.count(i) > 1})
print("offset pagination")
print("  page 1 ids:", [item["id"] for item in page1["items"]])
print("  page 2 ids:", [item["id"] for item in page2["items"]])
print(f"  read {len(seen)} items, {len(set(seen))} distinct; duplicated: {duplicates}")

# --- Cursor pagination: "give me the ten after the last one I saw".
seen = []
page1 = requests.get(f"{BASE}/feed", params={"cursor": 10**9, "limit": LIMIT},
                     timeout=5).json()
seen += [item["id"] for item in page1["items"]]
publish(3)
page2 = requests.get(f"{BASE}/feed", params={"cursor": page1["next_cursor"],
                                             "limit": LIMIT}, timeout=5).json()
seen += [item["id"] for item in page2["items"]]

print("\ncursor pagination")
print("  page 1 ids:", [item["id"] for item in page1["items"]])
print("  page 2 ids:", [item["id"] for item in page2["items"]])
print(f"  read {len(seen)} items, {len(set(seen))} distinct; "
      f"consecutive: {seen == list(range(seen[0], seen[0] - len(seen), -1))}")
