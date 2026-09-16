# A 200 is not a promise about the body. Check the shape of what came back before
# anything downstream relies on it.

import requests

from meridian_api import serve

BASE = serve()


def naive_total(body: dict) -> float:
    return sum(region.get("revenue", 0) for region in body["regions"])


def check_revenue_body(body: dict) -> list[str]:
    """Every way this body differs from what the code below it assumes."""
    problems = []
    if not isinstance(body.get("regions"), list):
        return ["'regions' is missing or not a list"]
    for i, region in enumerate(body["regions"]):
        for field, kind in (("region", str), ("revenue", (int, float))):
            if field not in region:
                problems.append(f"regions[{i}] has no '{field}' (keys: {sorted(region)})")
            elif not isinstance(region[field], kind):
                problems.append(f"regions[{i}].{field} is {type(region[field]).__name__}, "
                                f"expected {kind if isinstance(kind, type) else 'number'}")
        if len(problems) >= 3:
            problems.append("...")
            break
    return problems


for path in ("/revenue", "/revenue-v2"):
    response = requests.get(f"{BASE}{path}", params={"year": 2024, "quarter": 3}, timeout=5)
    body = response.json()
    print(f"{path}: status {response.status_code}")
    try:
        print(f"  naive total: {naive_total(body):,.2f}")
    except TypeError as error:
        print(f"  naive total: TypeError: {error}")
    problems = check_revenue_body(body)
    print("  shape check:", "ok" if not problems else "FAILED")
    for problem in problems:
        print("   -", problem)
