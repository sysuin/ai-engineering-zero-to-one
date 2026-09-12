# timeout: 1800
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# Reads code/21/_scorecard.json, written by 01. Re-run that first if the golden
# set or the scorer changes — this file grades the answers 01 recorded.
# Two annotators, one set of answers, and why they disagree.

import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                           # noqa: E402
from clarity.evals.runner import load                           # noqa: E402
from openai import OpenAI                                       # noqa: E402

client = OpenAI()
cases = {c["id"]: c for c in load()}
rows = [r for r in json.load(open("code/21/_scorecard.json"))
        if r["kind"] != "unanswerable"]

# Two rubrics that a reasonable person would call the same rubric. The difference is
# one clause each, and that is the point: nobody thinks they are disagreeing.
STRICT = ("You grade answers about Meridian. Reply CORRECT or WRONG and nothing "
          "else.\n"
          "CORRECT means the answer states the expected fact.\n"
          "An answer that states the fact but also adds unrequested material, or "
          "hedges, or answers a slightly wider question, is WRONG.")
LENIENT = ("You grade answers about Meridian. Reply CORRECT or WRONG and nothing "
           "else.\n"
           "CORRECT means a careful reader would come away knowing the expected "
           "fact.\n"
           "Extra detail, hedging, or a fuller answer than asked for is fine.")


def grade(rubric: str, row: dict) -> int:
    case = cases[row["id"]]
    reply = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=16,
        messages=[{"role": "system", "content": rubric},
                  {"role": "user",
                   "content": f"Question: {case['question']}\n"
                              f"Expected: {case['answer']}\n"
                              f"Answer: {row['answer'][:900]}"}])
    return int("CORRECT" in (reply.choices[0].message.content or "").upper())


with ThreadPoolExecutor(max_workers=8) as pool:
    strict = list(pool.map(lambda r: grade(STRICT, r), rows))
    lenient = list(pool.map(lambda r: grade(LENIENT, r), rows))
mechanical = [int(r["correct"]) for r in rows]


def kappa(a: list[int], b: list[int]) -> tuple[float, float, float]:
    """
    Cohen's kappa: agreement above what two people would reach by guessing.

    Raw agreement is the number everybody quotes and it is nearly useless when one
    label dominates. If 84% of answers are correct, two annotators who say CORRECT to
    everything agree 84% of the time and have learned nothing.
    """
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return observed, expected, (observed - expected) / (1 - expected)


print(f"{len(rows)} answers, graded by two rubrics that differ by one clause.\n")
for name, votes in (("strict rubric", strict), ("lenient rubric", lenient),
                    ("the string rule", mechanical)):
    print(f"  {name:<18} says correct {sum(votes):>3} of {len(votes)}  "
          f"({sum(votes) / len(votes):.0%})")

observed, expected, k = kappa(strict, lenient)
print()
print(f"  raw agreement between the two annotators   {observed:.0%}")
print(f"  agreement expected from chance alone       {expected:.0%}")
print(f"  Cohen's kappa                              {k:.2f}")

pairs = {"strict vs string rule": kappa(strict, mechanical),
         "lenient vs string rule": kappa(lenient, mechanical)}
for name, (obs, _, kk) in pairs.items():
    print(f"  {name:<42} agreement {obs:.0%}, kappa {kk:.2f}")

json.dump({"strict": strict, "lenient": lenient, "mechanical": mechanical,
           "kappa": k, "observed": observed, "expected": expected,
           "ids": [r["id"] for r in rows]},
          open("code/21/_agreement.json", "w"), indent=1)

split = [i for i in range(len(rows)) if strict[i] != lenient[i]]
print()
print(f"They disagreed on {len(split)} of {len(rows)}. Two of them:\n")
for i in split[:2]:
    flat = " ".join(rows[i]["answer"].split())
    print(f"  Q: {rows[i]['question'][:64]}")
    print(f"  A: {flat[:70]}")
    print(f"     expected {cases[rows[i]['id']]['answer'][:30]!r}  ->  "
          f"strict {'CORRECT' if strict[i] else 'WRONG'}, "
          f"lenient {'CORRECT' if lenient[i] else 'WRONG'}")
    print()
print("These are stand-ins for two people, and the arithmetic is identical when the")
print("annotators are human — as is the cause. Neither rubric is wrong. They encode")
print("different answers to a question nobody asked out loud: does a fuller answer")
print("than requested still count?")
print()
print("That question has to be settled before annotation, not discovered during it.")
print("Every hour spent making a rubric decidable is an hour you do not spend")
print("arguing about labels you have already paid for.")
