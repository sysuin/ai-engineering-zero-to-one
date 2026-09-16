# timeout: 300
# A provider's rate limit is a capacity plan somebody else wrote for you. Read it as one.

import json
import math

# Placeholders, like the rates: a limit on requests and on tokens per minute, of the size
# a provider might grant one account. Yours are on your provider's account page.
LIMIT_RPM = 5_000
LIMIT_TPM = 2_000_000

envelope = json.load(open("code/31/_envelope.json"))["designs"]


def minute_budget(tokens_per_call: int, seconds_per_call: float) -> dict:
    """What one minute of the limit buys, and how many workers it takes to use it."""
    by_tokens = LIMIT_TPM / tokens_per_call
    by_requests = LIMIT_RPM
    calls = min(by_tokens, by_requests)
    return {"calls_per_minute": calls,
            "binding": "tokens" if by_tokens < by_requests else "requests",
            # Little's law once more: workers in flight = throughput x duration.
            "workers": calls / 60 * seconds_per_call}


# ------------------------------------------------------------------ the nightly enricher
nightly = envelope["a nightly contract-risk enricher"]
CALLS = 100_000                        # 50,000 contracts x 2 calls
IN, OUT, SECONDS, WINDOW = 6_000, 400, 2.0, 8
print("The nightly enricher: 100,000 calls, 8 hours to make them\n")
for allowed_output in (OUT, 4_000):
    # Some providers count the output a request is *allowed*, not the output it produced,
    # when deciding whether to admit it. A generous max_tokens then costs capacity.
    plan = minute_budget(IN + allowed_output, SECONDS)
    hours = CALLS / plan["calls_per_minute"] / 60
    fits = "fits" if hours <= WINDOW else "does not fit"
    print(f"  counting {IN + allowed_output:,} tokens a call "
          f"(max output {allowed_output:,})")
    print(f"    {plan['calls_per_minute']:,.0f} calls a minute, limited by "
          f"{plan['binding']}; {plan['workers']:.1f} workers use all of it")
    print(f"    {hours:.1f} hours to finish: {fits} in {WINDOW}, "
          f"{(WINDOW - hours) / WINDOW:+.0%} margin")
    print()

print("  Workers beyond that number do not finish sooner. They collect 429s, and their")
print("  retries spend the limit that the workers doing real work needed.\n")

# ------------------------------------------------------------------ the meeting summaries
meetings = envelope["meeting summaries inside a video product"]
PER_DAY, CALLS_EACH, TOKENS, PEAK = 200_000, 2, 12_600, 3.0
average = PER_DAY * CALLS_EACH / 1440
busiest = average * PEAK
capacity = minute_budget(TOKENS, meetings["latency"] / CALLS_EACH)["calls_per_minute"]
print("Meeting summaries: 400,000 calls a day, arriving unevenly\n")
print(f"  {'':<26} {'calls/min':>10} {'tokens/min':>12} {'x the limit':>12}")
for label, calls in (("average minute", average), ("busiest hour", busiest)):
    tokens = calls * TOKENS
    print(f"  {label:<26} {calls:>10,.0f} {tokens:>12,.0f} {tokens / LIMIT_TPM:>11.1f}x")
print(f"  {'the limit admits':<26} {capacity:>10,.0f} {LIMIT_TPM:>12,}")

backlog = (busiest - capacity) * 60
print()
print(f"  During the busiest hour the queue grows by {backlog:,.0f} calls. Even the")
print(f"  average minute needs {average * TOKENS / LIMIT_TPM:.1f}x the limit, so the "
      "queue never drains:")
print("  smoothing arrivals cannot help when the average itself is over. The")
print(f"  design needs about {math.ceil(busiest * TOKENS / LIMIT_TPM)}x this limit at the "
      "busiest hour: a higher tier, capacity")
print("  reserved with the provider, or a batch path with its own quota for summaries")
print("  nobody reads within the hour — decided now, not discovered on launch day.")

json.dump({"limit_tpm": LIMIT_TPM, "limit_rpm": LIMIT_RPM,
           "meetings_average_x": average * TOKENS / LIMIT_TPM,
           "meetings_busiest_x": busiest * TOKENS / LIMIT_TPM},
          open("code/31/_rate_limits.json", "w"), indent=1)
