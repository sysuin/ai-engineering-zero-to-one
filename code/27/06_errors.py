# Errors are part of the interface. Three things an application can do about a failed
# model call — retry it, send it elsewhere, or give up — and which errors belong to which.

import sys

import httpx
import openai

sys.path.insert(0, "code")
from clarity.platform.gateway import Gateway, Reply              # noqa: E402

REQUEST = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")


def http_error(kind, status, message, headers=None, code=None):
    response = httpx.Response(status, request=REQUEST, headers=headers or {})
    return kind(message, response=response, body={"code": code} if code else None)


ERRORS = [
    ("rate limited", http_error(openai.RateLimitError, 429, "slow down",
                                {"retry-after": "7"})),
    ("out of credit", http_error(openai.RateLimitError, 429, "quota exceeded",
                                 code="insufficient_quota")),
    ("server error", http_error(openai.InternalServerError, 500, "internal")),
    ("overloaded", http_error(openai.InternalServerError, 503, "overloaded")),
    ("timed out", openai.APITimeoutError(request=REQUEST)),
    ("connection refused", openai.APIConnectionError(request=REQUEST)),
    ("prompt too long", http_error(openai.BadRequestError, 400, "context too long",
                                   code="context_length_exceeded")),
    ("malformed request", http_error(openai.BadRequestError, 400, "invalid schema")),
    ("bad key", http_error(openai.AuthenticationError, 401, "invalid key")),
    ("no access to model", http_error(openai.NotFoundError, 404, "model not found")),
]


def classify(error: Exception) -> tuple[str, str]:
    """What the caller should do, and why. The whole policy, in one place."""
    if isinstance(error, (openai.APITimeoutError, openai.APIConnectionError)):
        return "retry", "transient; fall back if it persists"
    if isinstance(error, openai.RateLimitError):
        # One status, two meanings: too fast, or out of money. Retrying the second
        # is a loop that ends when someone pays the bill.
        if (error.body or {}).get("code") == "insufficient_quota":
            return "fail", "no retry will pay the bill"
        wait = error.response.headers.get("retry-after")
        return "retry", f"after the {wait}s it asked for" if wait else "with backoff"
    if isinstance(error, openai.InternalServerError):
        return "retry", "the provider's fault; then fall back"
    if isinstance(error, openai.BadRequestError):
        if (error.body or {}).get("code") == "context_length_exceeded":
            return "shrink", "trim it, or route to a larger window"
        return "fail", "every provider will refuse it"
    if isinstance(error, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "fail", "configuration: page a person"
    if isinstance(error, openai.NotFoundError):
        return "fall back", "another provider may serve it"
    return "fail", "unknown: do not guess"


print(f"  {'error':<20} {'status':>6}  {'action':<10} why")
for name, error in ERRORS:
    status = getattr(getattr(error, "response", None), "status_code", "-")
    action, why = classify(error)
    print(f"  {name:<20} {status:>6}  {action:<10} {why}")


# ------------------------------------------------------------------ what the gateway does
class Failing:
    def __init__(self, name, error):
        self.name, self.error, self.calls = name, error, 0

    def complete(self, messages, **kwargs) -> Reply:
        self.calls += 1
        raise self.error


malformed = dict(ERRORS)["malformed request"]
primary, backup = Failing("primary", malformed), Failing("backup", malformed)
try:
    Gateway(primary, [backup]).complete([{"role": "user", "content": "hi"}])
except RuntimeError:
    pass
print(f"\nClarity's gateway on a malformed request: primary called {primary.calls}, "
      f"backup called {backup.calls}.")
print("It falls back on every exception, so a request no provider will accept is sent")
print("to each of them in turn, and its real cause arrives inside a RuntimeError.")
