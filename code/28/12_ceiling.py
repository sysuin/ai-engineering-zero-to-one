# Depends on code/32/_card.json; re-run after Chapter 32's numbers card changes.
# A per-request token ceiling, set from the distribution rather than guessed. The 120 answers
# behind the numbers card, replayed under each ceiling on one stated assumption: a request that
# reaches the ceiling stops there and refuses. No model calls.

import json

import numpy as np

rows = json.load(open("code/32/_card.json"))["rows"]
tokens = np.array([r["tokens"] for r in rows])
UNANSWERABLE = "unanswerable"


def replay(ceiling: float) -> dict:
    spent = right = cut = cut_right = rescued = 0
    for r in rows:
        if r["tokens"] <= ceiling:
            spent += r["tokens"]
            right += r["correct"]
            continue
        cut += 1
        spent += ceiling
        # A refusal is the right outcome only for a question with no answer.
        now_right = r["kind"] == UNANSWERABLE
        right += now_right
        cut_right += r["correct"] and not now_right
        rescued += now_right and not r["correct"]
    return {"spent": spent, "right": right, "cut": cut,
            "lost": cut_right, "rescued": rescued}


base = replay(float("inf"))
print(f"{len(rows)} answers, {base['right']} right, {base['spent']:,} tokens with no ceiling\n")
print(f"  {'ceiling':<14}{'cut':>5}{'unanswerable':>14}{'tokens saved':>14}{'right':>7}"
      f"{'lost':>6}{'rescued':>9}")
for label, ceiling in [(f"p{q} ({np.percentile(tokens, q):,.0f})", np.percentile(tokens, q))
                       for q in (50, 75, 90, 95, 99)] + [("none", float("inf"))]:
    out = replay(ceiling)
    unanswerable = sum(r["tokens"] > ceiling and r["kind"] == UNANSWERABLE for r in rows)
    saved = 1 - out["spent"] / base["spent"]
    print(f"  {label:<14}{out['cut']:>5}{unanswerable:>14}{saved:>14.0%}{out['right']:>7}"
          f"{out['lost']:>6}{out['rescued']:>9}")

print("\n  unanswerable = how many of the cut requests had no answer to find;")
print("  lost = right answers the ceiling cut off; rescued = unanswerable questions")
print("  answered wrongly that the forced refusal would have fixed")
