# The six status codes you will actually meet, each produced on purpose.

import requests

from meridian_api import serve

BASE = serve()

CALLS = [
    ("200 the happy path",      "/revenue?year=2024&quarter=3"),
    ("400 you asked wrongly",   "/revenue"),
    ("401 who are you",         "/account"),
    ("404 no such thing",       "/revenue?year=2031&quarter=1"),
    ("429 slow down",           "/ratelimited"),
    ("500 our fault",           "/boom"),
]

for label, path in CALLS:
    response = requests.get(BASE + path, timeout=10)
    body = response.json()
    detail = body.get("message") or f"total_revenue = {body['total_revenue']:,.2f}"
    print(f"{response.status_code}  {label:24} {detail}")

print()
print("The first digit is the whole story:")
print("  2xx  it worked")
print("  4xx  you did something wrong  -> fix the request, do not retry")
print("  5xx  they did something wrong -> retry, carefully")
