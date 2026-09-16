# timeout: 900
# How many near-identical excerpts can sit beside the right one before the answer goes wrong?
# The weakest-region commentary never names its quarter, so only the excerpt's label can.

import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST               # noqa: E402
from meridian_index import load_index               # noqa: E402

client = OpenAI()
chunks, _ = load_index()
COMMENTARY = {c["source"].removesuffix(".md").removeprefix("qbr-"): c for c in chunks
              if c["kind"] == "quarterly-review" and c["heading"] == "Commentary"}
QUARTERS = sorted(COMMENTARY)                        # "2023-Q1" ... "2025-Q4"


def weakest_figure(chunk) -> str:
    return re.search(r"weakest at \$([\d,]+)", chunk["text"]).group(1)


SYSTEM = ("Answer only from the excerpts. If they do not contain the answer, reply exactly: "
          "NOT IN THE DOCUMENTS.")


UNCLEAR = []


def ask(quarter: str, distractors: int, labelled: bool) -> str:
    rng = random.Random(f"{quarter}-{distractors}")
    others = rng.sample([q for q in QUARTERS if q != quarter], distractors)
    shown = [quarter] + others
    rng.shuffle(shown)
    excerpts = "\n\n".join(
        (f"[qbr-{q}.md, Commentary]\n" if labelled else "[excerpt]\n") + COMMENTARY[q]["text"]
        for q in shown)
    year, qn = quarter.split("-")
    reply = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=300,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"<excerpts>\n{excerpts}\n</excerpts>\n\n"
                   f"Which region was weakest in {year} {qn}, and what was its revenue?"}],
    ).choices[0].message.content or ""
    if weakest_figure(COMMENTARY[quarter]) in reply:
        return "right"
    if any(weakest_figure(COMMENTARY[q]) in reply for q in others):
        return "wrong quarter"
    if "NOT IN THE DOCUMENTS" in reply:
        return "abstained"
    UNCLEAR.append((quarter, distractors, labelled, " ".join(reply.split())[:150]))
    return "other"


CONDITIONS = [(n, labelled) for labelled in (True, False) for n in (0, 3, 11)]
with ThreadPoolExecutor(max_workers=12) as pool:
    grid = {c: list(pool.map(lambda q, c=c: ask(q, *c), QUARTERS)) for c in CONDITIONS}

print(f"{len(QUARTERS)} questions per row, one per quarter; the answer is the weakest region's revenue\n")
print(f"  {'excerpts labelled with':24}{'distractors':>12}{'right':>8}{'wrong quarter':>15}"
      f"{'abstained':>11}{'other':>7}")
for (n, labelled), outcomes in grid.items():
    label = "source and section" if labelled else "nothing"
    print(f"  {label:24}{n:>12}" + "".join(f"{outcomes.count(o):>{w}}" for o, w in
          (("right", 8), ("wrong quarter", 15), ("abstained", 11), ("other", 7))))

for quarter, n, labelled, reply in UNCLEAR:
    print(f"\n'other' ({quarter}, {n} distractors, {'labelled' if labelled else 'unlabelled'}): {reply!r}")
