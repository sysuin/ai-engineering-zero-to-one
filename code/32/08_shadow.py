# timeout: 2400
# Shadowing a change: the candidate answers every request beside the live version, and only the
# live answer is shown. Every request becomes a pair. The golden set's 120 questions stand in for
# traffic; the live version runs twice, because a version also disagrees with itself.

import json
import math
import statistics
import sys
from statistics import NormalDist
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.evals.runner import correct, load                  # noqa: E402
from clarity.v0_6.retrieve import Retriever                     # noqa: E402
from clarity.v0_7.warehouse import Warehouse                    # noqa: E402
from clarity.v0_8.tools import build_tools                      # noqa: E402
from clarity.v0_9.agent import Agent, Budget                    # noqa: E402
from clarity.v1_0.clarity import SYSTEM                         # noqa: E402
from meridian_index import load_index                           # noqa: E402

cases = load()
chunks, vectors = load_index()
agent = Agent(build_tools(Retriever(chunks, vectors), Warehouse()), budget=Budget(steps=6))
VERSIONS = {"live": SYSTEM,
            "candidate": SYSTEM + " Answer in one sentence."}


def run_all(system: str) -> list:
    with ThreadPoolExecutor(max_workers=12) as pool:
        return list(pool.map(lambda c: agent.run(c["question"], system=system), cases))


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact p-value for b pairs that went one way and c the other."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


runs = {"live, first run": run_all(VERSIONS["live"]),
        "live, second run": run_all(VERSIONS["live"]),
        "candidate": run_all(VERSIONS["candidate"])}
right = {name: [correct(c, r.answer) for c, r in zip(cases, rs)] for name, rs in runs.items()}

print(f"{len(cases)} questions; the candidate adds one instruction: answer in one sentence\n")
print(f"  {'run':<18}{'right':>8}{'median tokens':>15}")
for name, rs in runs.items():
    tokens = statistics.median(r.tokens for r in rs)
    print(f"  {name:<18}{sum(right[name]):>5}/{len(cases)}{tokens:>15,.0f}")

print(f"\n  {'pair':<36}{'disagree':>9}{'lost':>6}{'gained':>8}{'p':>7}")
report = {}
for label, other in (("live against itself", "live, second run"),
                     ("live against the candidate", "candidate")):
    first = right["live, first run"]
    lost = sum(a and not b for a, b in zip(first, right[other]))
    gained = sum(b and not a for a, b in zip(first, right[other]))
    p = mcnemar(lost, gained)
    print(f"  {label:<36}{lost + gained:>9}{lost:>6}{gained:>8}{p:>7.2f}")
    report[label] = {"lost": lost, "gained": gained, "p": p}

print("\n  disagree = right in one run and wrong in the other; lost and gained count")
print("  from the live version's first run; p = exact two-sided McNemar test")


def flipped(other: str) -> set[str]:
    return {c["id"] for c, a, b in zip(cases, right["live, first run"], right[other]) if a != b}


noise, change = flipped("live, second run"), flipped("candidate")
print(f"\n  {len(noise & change)} of the {len(change)} questions the candidate changed also changed "
      "between the")
print("  live version's own two runs")
# How many requests would settle a difference of this size, paired or not (80% power, 5% alpha)?
z_alpha, z_power = NormalDist().inv_cdf(0.975), NormalDist().inv_cdf(0.8)
live = sum(right["live, first run"]) / len(cases)
cand = sum(right["candidate"]) / len(cases)
discordant = (report["live against the candidate"]["lost"]
              + report["live against the candidate"]["gained"]) / len(cases)
delta = abs(live - cand)
if delta > 0:
    paired = ((z_alpha * math.sqrt(discordant) + z_power * math.sqrt(discordant - delta ** 2))
              / delta) ** 2
    mean = (live + cand) / 2
    unpaired = 2 * ((z_alpha * math.sqrt(2 * mean * (1 - mean))
                     + z_power * math.sqrt(live * (1 - live) + cand * (1 - cand))) / delta) ** 2
    print(f"\nto tell {live:.1%} from {cand:.1%} with {discordant:.1%} of requests disagreeing:")
    print(f"  shadowed, every request a pair:      {math.ceil(paired):>6,} requests")
    print(f"  split between versions (A/B):        {math.ceil(unpaired):>6,} requests")
json.dump({"right": right, **report, "flipped_alone": sorted(noise), "flipped_candidate":
           sorted(change)}, open("code/32/_shadow.json", "w"), indent=2)
