# timeout: 900
# JSON mode guarantees JSON. How many different shapes does it choose across forty contracts,
# asked the same question? Level 3 of this chapter's first listing, run on every contract twice.

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()
ASK = ("Extract the payment terms and price-adjustment terms from this supply "
       "agreement.\n\n{text}")


def json_mode(text: str) -> dict:
    raw = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=300,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": ASK.format(text=text) +
                   "\n\nReply with a JSON object."}],
    ).choices[0].message.content
    return json.loads(raw)


def paths(value, prefix: str = "") -> frozenset[str]:
    """Every key path in an object, ignoring values: its shape."""
    if isinstance(value, dict):
        return frozenset().union(*[paths(v, f"{prefix}.{k}") for k, v in value.items()]) \
            or frozenset({prefix})
    if isinstance(value, list):
        return frozenset().union(*[paths(v, f"{prefix}[]") for v in value]) or frozenset({prefix})
    return frozenset({prefix})


texts = [c["text"] for c in contracts]
for run in (1, 2):
    with ThreadPoolExecutor(max_workers=10) as pool:
        records = list(pool.map(json_mode, texts))
    shapes = Counter(paths(r) for r in records)
    tops = Counter(frozenset(r) for r in records)
    common, count = shapes.most_common(1)[0]
    print(f"run {run}: {len(records)} contracts, all parsed")
    print(f"  distinct top-level key sets   {len(tops):>3}")
    print(f"  distinct full shapes          {len(shapes):>3}")
    print(f"  contracts sharing the commonest shape {count:>3}")
    if run == 1:
        print("  the commonest shape's key paths:")
        for path in sorted(common)[:6]:
            print(f"    {path}")
