# The retry that charges the customer twice.
#
# When a request times out you do not know whether it happened. Retrying a read is
# harmless. Retrying a write is a decision, and an idempotency key is how you make it
# safely.

import requests

from meridian_api import serve, reset

BASE = serve()
reset()

def create_order(sku: str, qty: int, key: str | None = None, lose_response: bool = False):
    headers = {"Idempotency-Key": key} if key else {}
    payload = {"sku": sku, "qty": qty, "simulate_lost_response": lose_response}
    try:
        response = requests.post(f"{BASE}/orders", json=payload,
                                 headers=headers, timeout=5)
        return response.status_code, response.json()
    except requests.exceptions.RequestException:
        return None, "no response — did it happen?"


print("Without an idempotency key")
print("  send:  ", create_order("MRD-CLE-001", 480, lose_response=True))
print("  retry: ", create_order("MRD-CLE-001", 480))
print("  orders on the server:", requests.get(f"{BASE}/orders", timeout=5).json()["count"])
print("  -> two orders. The customer receives 960 cloths.")

reset()

print("\nWith one")
print("  send:  ", create_order("MRD-CLE-001", 480, key="a3f1", lose_response=True))
print("  retry: ", create_order("MRD-CLE-001", 480, key="a3f1"))
print("  orders on the server:", requests.get(f"{BASE}/orders", timeout=5).json()["count"])
print("  -> one order, and the retry was told what the first one did.")
