# Did the launch work? A simulated year of minutes per ticket for two support teams, with a
# season that both teams share. Every number below is an assumption, written down here, so
# the point is the method: what each comparison measures when the true effect is known.

import json
import random
import statistics

WEEKS, LAUNCH = 52, 26               # the system goes live for team A in week 26
BASE_MINUTES = 6.0
NOISE = 0.35                         # week-to-week wobble in a team's average, in minutes


def season(week: int) -> float:
    # Busier, slower weeks in the first half of the year, easing towards the summer.
    return 1.2 * (1 - week / WEEKS)


def simulate(effect: float, rng: random.Random):
    a, b = [], []
    for w in range(WEEKS):
        shared = BASE_MINUTES + season(w)
        a.append(shared + (effect if w >= LAUNCH else 0) + rng.gauss(0, NOISE))
        b.append(shared + rng.gauss(0, NOISE))
    return a, b


def change(team: list[float]) -> float:
    after, before = team[LAUNCH:], team[:LAUNCH]
    return statistics.mean(after) - statistics.mean(before)


def diff_in_diff(a: list[float], b: list[float]) -> float:
    # Team A's change, minus team B's change without the system.
    return change(a) - change(b)


def interval(values: list[float]) -> tuple[float, float]:
    # A 95% interval for a mean of weeks (normal approximation).
    mean = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / len(values) ** 0.5
    return mean - half, mean + half


def did_interval(a, b):
    gaps = [x - y for x, y in zip(a, b)]
    before, after = gaps[:LAUNCH], gaps[LAUNCH:]
    diff = statistics.mean(after) - statistics.mean(before)
    half = 1.96 * (statistics.variance(before) / len(before)
                   + statistics.variance(after) / len(after)) ** 0.5
    return diff - half, diff + half


rng = random.Random(6)
a, b = simulate(effect=-0.5, rng=rng)
print("One year, the system truly saves 0.5 minutes a ticket from week 26\n")
print(f"  team A before launch                   {statistics.mean(a[:LAUNCH]):.2f} min")
print(f"  team A after launch                    {statistics.mean(a[LAUNCH:]):.2f} min")
print(f"  team B, no system, same weeks, change {change(b):+.2f} min")
print(f"\n  before-and-after for team A            {change(a):+.2f} min")
low, high = did_interval(a, b)
print(f"  difference-in-differences              {diff_in_diff(a, b):+.2f} min"
      f"  (95% interval {low:+.2f} to {high:+.2f})")

# Now a thousand years in which the system does nothing at all.
RUNS = 1_000
claims_ba = claims_did = 0
for _ in range(RUNS):
    a, b = simulate(effect=0.0, rng=rng)
    low, high = interval(a[LAUNCH:])
    before_low, before_high = interval(a[:LAUNCH])
    claims_ba += high < before_low              # after is clearly below before
    low, high = did_interval(a, b)
    claims_did += high < 0
print(f"\n{RUNS:,} simulated years in which the system saves nothing")
print(f"  before-and-after 'finds' a saving     {claims_ba / RUNS:>6.1%} of years")
print(f"  difference-in-differences finds one   {claims_did / RUNS:>6.1%} of years")

# The pilot team is chosen because its numbers were the worst. Its bad weeks were partly bad
# luck, and luck does not persist: the team improves after launch with no help from anyone.


def pilot_improves(pick_worst: bool) -> bool:
    teams = [simulate(effect=0.0, rng=rng)[0] for _ in range(6)]
    recent = [statistics.mean(t[LAUNCH - 8:LAUNCH]) for t in teams]
    chosen = recent.index(max(recent)) if pick_worst else rng.randrange(6)
    control = [statistics.mean(t[w] for i, t in enumerate(teams) if i != chosen)
               for w in range(WEEKS)]
    gaps = [x - y for x, y in zip(teams[chosen], control)]
    return statistics.mean(gaps[LAUNCH:]) < statistics.mean(gaps[LAUNCH - 8:LAUNCH]) - 0.2


print("\nSix teams, one pilot, a system that saves nothing. How often does the pilot team")
print("improve by more than 0.2 minutes against the other five, measured from its last")
print("eight weeks before launch?")
pilots = {}
for label, worst in (("pilot chosen at random", False), ("pilot chosen as the worst", True)):
    rate = sum(pilot_improves(worst) for _ in range(RUNS)) / RUNS
    print(f"  {label:<28} {rate:>6.1%} of years")
    pilots[label] = rate

json.dump({"before-and-after": claims_ba / RUNS, "difference-in-differences": claims_did / RUNS,
           **pilots}, open("code/06/_before_after.json", "w"), indent=1)
