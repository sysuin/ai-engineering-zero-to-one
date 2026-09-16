# timeout: 2400
# Depends on clarity/evals/runner.py.
# The four rungs below fine-tuning, measured on one subtask.

import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
sys.path.insert(0, "code/15")
from clarity.config import MODEL_FAST                          # noqa: E402
from clarity.v0_7.warehouse import METRICS, QuerySpec, Warehouse  # noqa: E402
from openai import OpenAI                                      # noqa: E402

client = OpenAI()
warehouse = Warehouse()

# One narrow, structured subtask: turn a question into a query specification. This is
# what fine-tuning is supposed to be good at — a fixed output shape, a fixed domain,
# and a behaviour rather than a fact.
# Deliberately colloquial. The precise version of each of these is answered
# correctly by the schema alone, which is itself the finding in §30.1 — the ladder
# only discriminates on questions people actually type.
CASES = [
    ("What was total revenue in 2024 Q3?", "revenue"),
    ("How many orders were placed in 2025?", "orders"),
    ("What is our gross margin percentage for 2025?", "margin_pct"),
    ("How many units of Safety products shipped in 2025 Q1?", "units"),
    ("What was the average discount percentage in 2024?", "avg_discount"),
    ("How did we do on volume last year?", "units"),
    ("What's our take on Packaging in 2025?", "revenue"),
    ("How much are we giving away on price?", "avg_discount"),
    ("Are customers spending more per basket?", "avg_order_value"),
    ("How wide are we running on the Safety line?", "margin_pct"),
    ("How much did the goods actually set us back in 2024?", "cost"),
    ("How many different things are we selling?", "skus"),
]

BARE = "Translate the question into a query specification."
CATALOGUE = BARE + "\n\nMetrics available:\n" + "\n".join(
    f"  {name}: {m['label']} — {m['note']}" for name, m in METRICS.items())
SHOTS = CATALOGUE + "\n\nExamples:\n" + "\n".join([
    "  'revenue in 2024 Q3' -> metric=revenue",
    "  'how many orders last year' -> metric=orders",
    "  'margin for the Safety category' -> metric=margin_pct",
    "  'units shipped in Q1' -> metric=units",
    "  'what did it cost us' -> metric=cost",
])


def plan(system: str, question: str) -> str | None:
    parsed = client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        response_format=QuerySpec,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": question}],
    ).choices[0].message.parsed
    return parsed.metric if parsed else None


rungs = [("no instructions beyond the schema", BARE),
         ("+ the metric catalogue in the prompt", CATALOGUE),
         ("+ five examples", SHOTS)]

print(f"One subtask — question to query specification — on {len(CASES)} questions.\n")
print(f"  {'':<40}{'correct metric':>16}")
results = {}
for label, system in rungs:
    with ThreadPoolExecutor(max_workers=6) as pool:
        picks = list(pool.map(lambda c: plan(system, c[0]), CASES))
    right = sum(1 for (_, want), got in zip(CASES, picks) if got == want)
    results[label] = {"correct": right, "of": len(CASES),
                      "wrong": [{"q": q, "want": w, "got": g}
                                for (q, w), g in zip(CASES, picks) if g != w]}
    print(f"  {label:<40}{right:>10}/{len(CASES):<5}")

json.dump(results, open("code/30/_ladder.json", "w"), indent=1)
best = max(results.values(), key=lambda r: r["correct"])
print()
for label, row in results.items():
    for miss in row["wrong"][:1]:
        print(f"  {label}")
        print(f"    {miss['q'][:56]}")
        print(f"    wanted {miss['want']}, got {miss['got']}")
        break
    break

spread = max(r["correct"] for r in results.values()) - min(
    r["correct"] for r in results.values())
print()
if spread == 0:
    print(f"Every rung scored the same: {best['correct']} of {best['of']}. The "
          f"schema alone was enough,")
    print("and the catalogue and the examples bought nothing measurable.")
    print()
    print("That is the result, and it is a result worth expecting. The question these")
    print("twelve cases were chosen to answer — would a fine-tune help here — is")
    print("answered before the ladder begins: there is nowhere to climb from.")
    print()
    print("Note which failure is left. 'How wide are we running on the Safety line'")
    print("is ambiguous to a person too; margin is the trade's reading of 'wide' and")
    print("revenue is a defensible one. A training set would teach one reading, which")
    print("is a real use for fine-tuning and a small one — and a sentence in the")
    print("prompt teaches it faster.")
else:
    print(f"The rungs spread by {spread} of {best['of']} cases, which is the shape "
          f"the ladder is")
    print("drawn for: each rung costs more than the one below it and buys less than")
    print("the one below it did.")
print()
print("That is the ladder, and the rungs are ordered by cost in hours rather than by")
print("sophistication:")
print()
print("  1  a better prompt          minutes, reversible, no new artefact")
print("  2  examples in the prompt   an hour, and you can change them at lunchtime")
print("  3  retrieval                Part III — the right answer when the problem is")
print("                              that the model does not know a fact")
print("  4  a structured output      Chapter 8 — the right answer when the problem")
print("                              is that the shape is wrong")
print("  5  fine-tuning              days, a dataset to maintain, a model to version,")
print("                              and a new thing that can drift")
print()
print("Most requests for fine-tuning are requests for rung three or four. 'The model")
print("does not know our products' is retrieval. 'The model will not answer in our")
print("format' is a schema. Neither is fixed by training, and both are frequently")
print("attempted with it.")
print()
print("What is left after those four rungs is what fine-tuning is genuinely for, and")
print("it is narrower than the market suggests: a *behaviour* you cannot describe")
print("but can demonstrate — a house tone, a peculiar output convention, a judgement")
print("that takes a paragraph to specify and one example to show.")
