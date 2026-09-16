# timeout: 1800
# Depends on clarity/evals/judge.py.
# Ask the same judge the same question five times. Does it give the same verdict? And does
# a panel of its own verdicts do better than one?

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

sys.path.insert(0, "code")
sys.path.insert(0, "code/22")
from _figures import build                                      # noqa: E402
from clarity.config import MODEL_FAST                           # noqa: E402
from clarity.evals.judge import REFERENCE                       # noqa: E402

REPEATS = 5
client = OpenAI()
rows = [r for r in build() if r["form"] == "thousands"]    # the harder form


def verdict(row: dict, temperature: float | None) -> int:
    options = {} if temperature is None else {"temperature": temperature}
    reply = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=220, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": REFERENCE},
                  {"role": "user", "content": f"Question: {row['question']}\n\n"
                   f"Expected answer: {row['expected']}\n\nAnswer: {row['answer']}"}],
        **options).choices[0].message.content or "{}"
    try:
        return int("CORRECT" in str(json.loads(reply).get("verdict", "")).upper())
    except json.JSONDecodeError:
        return 0


print(f"{len(rows)} answers, each judged {REPEATS} times\n")
print(f"  {'':26} {'unanimous':>9} {'single right':>13} {'majority right':>15}")
for label, temperature in (("default temperature", None), ("temperature 0", 0.0)):
    jobs = [(row, temperature) for row in rows for _ in range(REPEATS)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        votes = list(pool.map(lambda job: verdict(*job), jobs))
    per_row = [votes[i * REPEATS:(i + 1) * REPEATS] for i in range(len(rows))]
    unanimous = sum(len(set(v)) == 1 for v in per_row)
    single = sum(v[0] == r["label"] for v, r in zip(per_row, rows))
    majority = sum(Counter(v).most_common(1)[0][0] == r["label"]
                   for v, r in zip(per_row, rows))
    print(f"  {label:26} {unanimous:>5} of {len(rows):<3}{single:>9} of {len(rows):<3}"
          f"{majority:>11} of {len(rows)}")
