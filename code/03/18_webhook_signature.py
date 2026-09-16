# When an API calls YOU, anyone on the internet can pretend to be that API. A signature
# proves the message came from someone who holds the shared secret, and was not changed.

import hashlib
import hmac
import json
import time

SECRET = b"whsec-demo-not-a-real-secret"       # shared once, out of band; lives in .env


def sign(body: bytes, timestamp: int) -> str:
    message = f"{timestamp}.".encode() + body
    return hmac.new(SECRET, message, hashlib.sha256).hexdigest()


def verify(body: bytes, timestamp: int, signature: str, now: int, tolerance=300) -> str:
    if abs(now - timestamp) > tolerance:
        return "rejected: too old (a replay?)"
    expected = sign(body, timestamp)
    # compare_digest takes the same time whether the first or the last character differs,
    # so an attacker cannot discover the signature one character at a time.
    if not hmac.compare_digest(expected, signature):
        return "rejected: signature does not match"
    return "accepted"


now = int(time.time())
body = json.dumps({"event": "batch.completed", "batch_id": "b_123"}).encode()
signature = sign(body, now)
print("signature:", signature[:16] + "...")

print("genuine:          ", verify(body, now, signature, now))
tampered = body.replace(b"b_123", b"b_999")
print("body changed:     ", verify(tampered, now, signature, now))
print("wrong secret:     ", verify(body, now, hmac.new(b"guess", body, hashlib.sha256)
                                   .hexdigest(), now))
print("replayed next day:", verify(body, now, signature, now + 86_400))
