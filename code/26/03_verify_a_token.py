# "Who is calling" is a signed token, and a token that is merely valid is a token from
# anywhere. Six tokens, checked two ways.

import time

import jwt

KEY, OTHER_KEY = "clarity-signing-key-" + "x" * 24, "someone-elses-key-" + "y" * 26
ISSUER, AUDIENCE = "https://auth.meridian.example", "clarity-api"
now = int(time.time())


def token(key=KEY, algorithm="HS256", **changes) -> str:
    claims = {"sub": "analyst-17", "tenant": "meridian", "role": "reader",
              "iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 900} | changes
    return jwt.encode(claims, key if algorithm != "none" else None, algorithm=algorithm)


def careless(raw: str) -> str:
    """Reads the claims without checking anything — the version that appears in demos."""
    claims = jwt.decode(raw, options={"verify_signature": False})
    return f"accepted as {claims['sub']}"


def careful(raw: str) -> str:
    try:
        claims = jwt.decode(raw, KEY, algorithms=["HS256"], audience=AUDIENCE, issuer=ISSUER,
                            options={"require": ["exp", "iat", "iss", "aud", "sub"]})
        return f"accepted as {claims['sub']}"
    except jwt.PyJWTError as error:
        return f"refused: {type(error).__name__}"


TOKENS = [
    ("a good token", token()),
    ("expired an hour ago", token(iat=now - 7200, exp=now - 3600)),
    ("issued for another API", token(aud="payroll-api")),
    ("from another issuer", token(iss="https://auth.example.org")),
    ("signed with someone else's key", token(key=OTHER_KEY)),
    ("no signature at all (alg none)", token(algorithm="none")),
]
print(f"  {'':34} {'claims read without checking':30} {'checked properly'}")
for label, raw in TOKENS:
    print(f"  {label:34} {careless(raw):30} {careful(raw)}")
