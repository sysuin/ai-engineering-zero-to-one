# What does this cost today? You cannot claim a saving without this number.
#
# And the number that decides everything is not accuracy. It is volume.

import json
from pathlib import Path

TICKETS = Path("data/meridian/documents/tickets/tickets.jsonl")
rows = [json.loads(line) for line in TICKETS.read_text().splitlines()]

# Three figures you get by asking, not by guessing. From Meridian's support lead:
MINUTES_TO_TRIAGE = 1.75      # read it, decide the category, route it
LOADED_HOURLY_RATE = 34.00    # salary plus employer costs
BUILD_DAYS = 12               # your honest estimate, doubled, as it always should be
DAY_RATE = 520.00


def annual_cost(tickets_per_year: float) -> float:
    return tickets_per_year * MINUTES_TO_TRIAGE / 60 * LOADED_HOURLY_RATE


print(f"The corpus holds {len(rows)} tickets. That is a sample, not a year.")
print(f"Triage takes {MINUTES_TO_TRIAGE} minutes at ${LOADED_HOURLY_RATE}/hour, so one")
print(f"ticket costs ${MINUTES_TO_TRIAGE / 60 * LOADED_HOURLY_RATE:.3f} to handle by hand.\n")

build_cost = BUILD_DAYS * DAY_RATE
print(f"Building it: {BUILD_DAYS} days at ${DAY_RATE:,.0f} = ${build_cost:,.0f}, "
      f"plus running costs.\n")

print(f"{'tickets/year':>14} {'manual cost':>13} {'saved at 80%':>14} "
      f"{'payback':>12}   verdict")
for per_year in (200, 2_000, 20_000, 200_000, 1_000_000):
    manual = annual_cost(per_year)
    saved = manual * 0.80
    payback = build_cost / saved if saved else float("inf")
    if payback > 5:
        verdict = "do not build"
    elif payback > 1:
        verdict = "marginal"
    else:
        verdict = "build it"
    payback_text = f"{payback:.1f} years" if payback < 100 else "never"
    print(f"{per_year:>14,} {manual:>13,.0f} {saved:>14,.0f} {payback_text:>12}   {verdict}")

print("\nSame task. Same accuracy. Same everything except how often it happens.")
print("Volume decides this, and volume is a question you can answer in an afternoon")
print("without writing a line of code.")
