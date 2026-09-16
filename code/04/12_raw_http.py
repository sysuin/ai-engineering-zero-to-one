# What the library sends. The same call as 01_first_call.py, made with `requests` and
# Chapter 3's vocabulary: a POST, a bearer token in a header, a JSON body.

import json

import requests

from clarity.config import MODEL_FAST, openai_key

key = openai_key()
body = {
    "model": MODEL_FAST,
    "messages": [{"role": "user", "content": "In one sentence, what is a supplier contract?"}],
    "max_completion_tokens": 60,
}

print("POST https://api.openai.com/v1/chat/completions")
print(f"Authorization: Bearer <your key, {len(key)} characters>")   # never print a key
print("Content-Type: application/json")
print(json.dumps(body, indent=2))

response = requests.post("https://api.openai.com/v1/chat/completions",
                         headers={"Authorization": f"Bearer {key}"},
                         json=body, timeout=(5, 60))

print(f"\nstatus {response.status_code}")
interesting = sorted(h for h in response.headers
                     if h.lower().startswith(("x-ratelimit", "x-request-id", "openai-")))
print("headers worth knowing about:")
for name in interesting:
    print(f"  {name}")

data = response.json()
print("\nbody keys:", ", ".join(data))
print("answer:   ", data["choices"][0]["message"]["content"])
print("usage:    ", {k: data["usage"][k] for k in ("prompt_tokens", "completion_tokens")})
