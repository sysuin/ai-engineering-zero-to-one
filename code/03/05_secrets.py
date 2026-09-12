# Where a key comes from, and where it must never be.

import os

import requests

from meridian_api import serve

BASE = serve()

# Read the key from the environment. The value is not in this file, is not in the
# repository, and cannot be committed by accident.
api_key = os.environ.get("MERIDIAN_API_KEY")

if not api_key:
    print("MERIDIAN_API_KEY is not set. Falling back to the demo key.")
    print("In a real project this should stop the program instead.\n")
    api_key = "meridian-demo-key"

# The key goes in a header. Never in the URL: URLs end up in server logs, in browser
# history, in error reports, and in the referrer of the next request.
response = requests.get(f"{BASE}/account",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=10)

print("With the key:", response.status_code, response.json())

no_key = requests.get(f"{BASE}/account", timeout=10)
print("Without it:  ", no_key.status_code, no_key.json()["message"])

# When you print anything for support, print a fingerprint, not the key.
print(f"\nSafe to paste into a ticket: key ending {api_key[-4:]}, length {len(api_key)}")
