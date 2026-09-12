# timeout: 1800
# The document fits. That is not the same as it working.
#
# One fact is planted at eleven depths in a long context, and asked for each time.

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tiktoken
from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
encoder = tiktoken.encoding_for_model(MODEL_FAST)

APPENDIX = Path("data/meridian/documents/awkward/03-long-appendix.md").read_text()
SECTIONS = [s for s in APPENDIX.split("## ") if s.strip()][1:]

NEEDLE = ("C.999 Depot override code\n\nThe emergency depot override code for the "
          "Columbus facility is QX-4471. It is revised annually.\n")
QUESTION = ("What is the emergency depot override code for the Columbus facility? "
            "Reply with the code only.")
ANSWER = "QX-4471"


def haystack(n_sections: int, depth: float) -> str:
    """Build a document of `n_sections` with the needle at `depth` through it."""
    body = SECTIONS[:n_sections]
    at = int(len(body) * depth)
    return "## " + "## ".join(body[:at] + [NEEDLE] + body[at:])


def ask(document: str) -> bool:
    answer = (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=32,
        messages=[{"role": "user",
                   "content": f"<document>\n{document}\n</document>\n\n{QUESTION}"}],
    ).choices[0].message.content or "")
    return ANSWER in answer


DEPTHS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SIZES = [40, 160, 400]

results = {}
print(f"{'context':>8} {'tokens':>7}  " +
      "".join(f"{int(d * 100):>4}%" for d in DEPTHS) + "  found")
for n in SIZES:
    documents = [haystack(n, d) for d in DEPTHS]
    tokens = len(encoder.encode(documents[0]))
    with ThreadPoolExecutor(max_workers=11) as pool:
        found = list(pool.map(ask, documents))
    results[tokens] = {str(d): f for d, f in zip(DEPTHS, found)}
    marks = "".join(f"{' yes' if f else '  NO'} " for f in found)
    print(f"{n:>5} sec {tokens:>7,}  {marks} {sum(found)}/{len(DEPTHS)}")

Path("code/10/_needle.json").write_text(json.dumps(results, indent=2))

print()
print("The fact is stated once, in plain language, in a document the model was given in")
print("full. Nothing was summarised, truncated or retrieved.")
print()
print("Whether it is found depends on where it sits — and the longer the context, the")
print("more that matters. 'It fits in the window' is a statement about the window, not")
print("about whether the model will use it.")
