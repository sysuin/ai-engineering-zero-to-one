# timeout: 1800
# Open-ended output, graded two ways: one holistic score, or a checklist of the facts a good
# summary must state. Summaries built from golden facts, so every flaw is known.

import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml

sys.path.insert(0, "code")
from openai import OpenAI                                          # noqa: E402

from clarity.config import MODEL_FAST                              # noqa: E402

client = OpenAI()
cases = {c["id"]: c for c in yaml.safe_load(open("code/clarity/evals/golden.yaml"))["cases"]}
QUARTERS = ["2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4", "2024-Q1", "2024-Q2", "2024-Q3",
            "2024-Q4"]


def facts(q: str) -> dict:
    get = lambda field: cases[f"qbr-{q}-{field}"]["answer"]           # noqa: E731
    return {"revenue": get("revenue"), "orders": get("orders"), "margin": get("margin"),
            "strongest": get("strongest"), "weakest": get("weakest")}


def summary(q: str, f: dict, variant: str) -> str:
    year, quarter = q.split("-")
    s = [f"In {year} {quarter} Meridian's revenue was ${f['revenue']} from {f['orders']} orders.",
         f"Gross margin was {f['margin']}.",
         f"{f['strongest']} was the strongest region and {f['weakest']} the weakest."]
    if variant == "omits the margin":
        del s[1]
    if variant == "regions swapped":
        s[2] = f"{f['weakest']} was the strongest region and {f['strongest']} the weakest."
    if variant == "padded":
        s += ["The quarter's figures come from the company's quarterly review.",
              "Regional performance is discussed in more detail in the review itself."]
    return " ".join(s)


VARIANTS = ["complete", "padded", "omits the margin", "regions swapped"]
GOOD = {"complete", "padded"}
REFERENCE = lambda f: "\n".join(f"- {k}: {v}" for k, v in f.items())        # noqa: E731


def holistic(ref: str, text: str) -> int:
    reply = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=300, response_format={"type": "json_object"},
        messages=[{"role": "system", "content":
                   'Grade a quarterly summary against the reference facts. Reply with JSON: '
                   '{"reason": "...", "score": 1 to 10}, reason first.'},
                  {"role": "user", "content": f"Reference facts:\n{ref}\n\nSummary: {text}"}])
    return int(json.loads(reply.choices[0].message.content or "{}").get("score", 0))


def checklist(ref: str, text: str) -> bool:
    reply = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=400, response_format={"type": "json_object"},
        messages=[{"role": "system", "content":
                   'For each reference fact, decide whether the summary states it correctly. '
                   'Reply with JSON: {"checks": [{"fact": "...", "stated_correctly": true or '
                   'false}]}. A fact that is missing is not stated correctly.'},
                  {"role": "user", "content": f"Reference facts:\n{ref}\n\nSummary: {text}"}])
    checks = json.loads(reply.choices[0].message.content or "{}").get("checks", [])
    return len(checks) >= 5 and all(c.get("stated_correctly") is True for c in checks)


items = [(q, v) for q in QUARTERS for v in VARIANTS]
with ThreadPoolExecutor(max_workers=10) as pool:
    scores = list(pool.map(lambda qv: holistic(REFERENCE(facts(qv[0])),
                                               summary(qv[0], facts(qv[0]), qv[1])), items))
    passes = list(pool.map(lambda qv: checklist(REFERENCE(facts(qv[0])),
                                                summary(qv[0], facts(qv[0]), qv[1])), items))

print(f"{len(QUARTERS)} quarters, four summaries each, built from the golden set's facts\n")
print(f"  {'summary':<20} {'holistic score':>15} {'checklist passed':>17}")
by_variant = {}
for variant in VARIANTS:
    s = [sc for (q, v), sc in zip(items, scores) if v == variant]
    p = [ok for (q, v), ok in zip(items, passes) if v == variant]
    by_variant[variant] = (s, p)
    print(f"  {variant:<20} {statistics.mean(s):>9.1f} ({min(s)}-{max(s)}) "
          f"{sum(p):>9} of {len(p)}")

good_min = min(min(by_variant[v][0]) for v in GOOD)
bad_max = max(max(by_variant[v][0]) for v in VARIANTS if v not in GOOD)
print(f"\n  lowest holistic score for a good summary: {good_min};  highest for a flawed one: "
      f"{bad_max}")
if bad_max >= good_min:
    print("  the ranges overlap, so no threshold on the score separates good from flawed")
else:
    print("  a threshold between them would separate good from flawed on this set")
checklist_right = sum(ok == (v in GOOD) for (q, v), ok in zip(items, passes))
print(f"  the checklist's pass/fail agreed with the truth on {checklist_right} of {len(items)}")
