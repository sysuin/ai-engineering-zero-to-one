# timeout: 900
# The primary is down. Does the answer arrive, and can you tell where from?

import json
import sys
import time

sys.path.insert(0, "code")
sys.path.insert(0, "code/27")
from _stub_provider import BlockTransport                      # noqa: E402
from clarity.config import MODEL_FAST                          # noqa: E402
from clarity.platform.gateway import (BlockShapedProvider,     # noqa: E402
                                      Gateway, OpenAIProvider)
from clarity.platform.resilience import CircuitBreaker, Open   # noqa: E402

MESSAGES = [{"role": "system", "content": "You are an analyst for Meridian."},
            {"role": "user", "content": "Say hello in five words."}]


class Broken:
    """A provider that is down, and honest about it."""
    name = "primary"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        time.sleep(0.2)                 # failures are not instant
        raise ConnectionError("503 from the primary")


rows = {}

print("--- no fallback ---\n")
primary = Broken()
gateway = Gateway(primary)
started = time.perf_counter()
try:
    gateway.complete(MESSAGES, max_tokens=40)
except RuntimeError as error:
    rows["no fallback"] = {"seconds": time.perf_counter() - started,
                           "answered": False,
                           "attempts": list(gateway.attempts)}
    print(f"  {rows['no fallback']['seconds']:.2f}s   "
          f"{str(error).splitlines()[0][:56]}")

print("\n--- with a fallback ---\n")
primary = Broken()
gateway = Gateway(primary, fallbacks=[OpenAIProvider(MODEL_FAST)])
started = time.perf_counter()
reply = gateway.complete(MESSAGES, max_tokens=40)
rows["with fallback"] = {"seconds": time.perf_counter() - started, "answered": True,
                         "attempts": list(gateway.attempts),
                         "answered_by": reply.provider}
print(f"  {rows['with fallback']['seconds']:.2f}s   answered by "
      f"{reply.provider}: {reply.text.strip()[:38]}")
print(f"  attempts: {' -> '.join(f'{n} ({s})' for n, s in gateway.attempts)}")

print("\n--- with a fallback and a breaker, over ten requests ---\n")
primary = Broken()
# One breaker per provider. A shared one is reset by the healthy fallback.
gateway = Gateway(primary, fallbacks=[OpenAIProvider(MODEL_FAST)],
                  guard_factory=lambda: CircuitBreaker(threshold=2, cooldown=5.0))
started = time.perf_counter()
for _ in range(10):
    gateway.complete(MESSAGES, max_tokens=40)
elapsed = time.perf_counter() - started
trips = gateway.guards["primary"].trips
rows["breaker over ten"] = {"seconds": elapsed, "primary_calls": primary.calls,
                            "trips": trips}
print(f"  {elapsed:.1f}s for ten requests")
print(f"  the dead primary was called {primary.calls} times, not 10")
print(f"  the breaker opened {trips} time(s)")

json.dump(rows, open("code/27/_fallback.json", "w"), indent=1)

print()
print("Three things are worth separating here, because they are usually conflated.")
print()
print("A fallback answers the question. Without one, a provider outage is your")
print("outage; with one, it is a line in a trace.")
print()
print("A breaker stops you paying for the outage. The dead primary was called twice")
print("rather than ten times, and the other eight requests went straight to the")
print("fallback without waiting for a timeout first.")
print()
print("And the record of which provider answered is not a nicety. An answer from a")
print("different model is a different answer — different refusal behaviour, different")
print("tool choices, different failure modes — and a trace that does not say which")
print("one produced it is a trace that cannot explain the day your quality dipped")
print("for ninety minutes.")
print()
print("Which leads to the uncomfortable part of fallbacks: you are promising that a")
print("second model is good enough to serve your users unattended. §27.3 is how you")
print("find out whether that is true, and the answer is not 'it is a good model'.")
