# timeout: 1200
# temperature=0 is not determinism, and the gap is where the flaky bugs live.

import json
import sys
from collections import Counter

sys.path.insert(0, "code")
from clarity.config import CAPABILITIES, MODEL_FAST            # noqa: E402
from openai import OpenAI                                      # noqa: E402

RUNS = 20
client = OpenAI()
QUESTION = ("A supplier may raise prices once in any twelve month period by no more "
            "than 3%, on 30 days notice. They raised prices 2.8% on 25 days notice, "
            "twice in one year. List every clause breached, most serious first.")


def ask(**extra) -> str:
    reply = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=300,
        messages=[{"role": "user", "content": QUESTION}], **extra)
    return (reply.choices[0].message.content or "").strip()


settings = [("default", {}), ("temperature=0", {"temperature": 0})]
if "seed" in CAPABILITIES.get(MODEL_FAST, set()):
    settings.append(("temperature=0 + seed", {"temperature": 0, "seed": 7}))

print(f"The same question, {RUNS} times, on {MODEL_FAST}.\n")
results = {}
for label, extra in settings:
    answers = [ask(**extra) for _ in range(RUNS)]
    counts = Counter(answers)
    longest = max(counts.values())
    results[label] = {"distinct": len(counts), "modal": longest,
                      "lengths": sorted({len(a) for a in answers})}
    print(f"  {label:<22}{len(counts):>2} distinct answers of {RUNS}   "
          f"most common appeared {longest}x")

json.dump(results, open("code/24/_determinism.json", "w"), indent=1)

best = min(results.values(), key=lambda r: r["distinct"])
print()
if best["distinct"] == 1:
    print("One setting did produce identical text every time. Do not build on it.")
else:
    print(f"The steadiest setting still produced {best['distinct']} different answers.")
print()
print("temperature=0 means 'always take the most likely token', and that is not the")
print("same as 'always produce the same text'. Floating-point addition is not")
print("associative, so a batch of a different size, a different GPU, or a different")
print("kernel gives a different sum; when two tokens are nearly tied, a small")
print("difference in the numbers decides which one wins, and every token after")
print("it is downstream of that choice.")
print()
seeded = results.get("temperature=0 + seed")
plain = results.get("temperature=0")
if seeded and plain:
    print(f"The seed changed nothing measurable here: {plain['distinct']} distinct "
          f"answers without it,")
    print(f"{seeded['distinct']} with. A seed fixes the sampling, not the arithmetic, "
          f"and it cannot fix a")
    print("provider that routed you to different hardware between two calls.")
print()
print("So the working assumption for the rest of this chapter is:")
print()
print("  the same input can produce a different output, on purpose-built")
print("  infrastructure, with every knob set to its most deterministic value")
print()
print("Which means 'I cannot reproduce it' is not a diagnosis. It is the normal")
print("condition, and §24.3 is about what you do instead.")
