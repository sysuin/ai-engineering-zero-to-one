# timeout: 600
# A streamed structured output, snapshot by snapshot. The SDK hands you a partially parsed
# object on every chunk; this counts how often a value it showed was not the value that
# arrived, and what one rule for "complete" costs.

import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jiter
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                      # noqa: E402
from clarity.prompts import load                           # noqa: E402
from clarity.v0_3.extract import QuarterlyReview           # noqa: E402

client = OpenAI()
REVIEWS = [Path(f"data/meridian/documents/quarterly-reviews/qbr-2024-Q{q}.md") for q in (1, 2, 3)]


def snapshots(path: Path) -> list[str]:
    """The raw JSON text after every chunk of one streamed extraction."""
    seen = []
    with client.chat.completions.stream(
            model=MODEL_FAST, max_completion_tokens=1200, response_format=QuarterlyReview,
            messages=[{"role": "system", "content": load("extract_review")},
                      {"role": "user", "content": f"<document>\n{path.read_text()}\n</document>"}],
    ) as stream:
        for event in stream:
            if event.type == "content.delta":
                seen.append(event.snapshot)
    return seen


def leaves(value, path: str = "") -> list[tuple[str, object]]:
    """Every scalar in a (partial) object, with its path, in the order it was written."""
    if isinstance(value, dict):
        return [leaf for k, v in value.items() for leaf in leaves(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [leaf for i, v in enumerate(value) for leaf in leaves(v, f"{path}[{i}]")]
    return [(path, value)]


def the_sdk(text: str):
    """What the stream's event.parsed holds."""
    if not text.strip():                 # the SDK skips these too
        return []
    return leaves(jiter.from_json(text.encode(), partial_mode=True))


def strings_as_they_arrive(text: str):
    """For a UI that wants text to appear as it is written."""
    if not text.strip():
        return []
    mode = "trailing-strings"
    return leaves(jiter.from_json(text.encode(), partial_mode=mode))


def numbers_once_ended(text: str):
    """A number is final once a character follows it."""
    values = the_sdk(text)          # no unfinished strings
    last = values[-1][1] if values else None
    still_typing = re.search(r"[\d.eE+-]$", text)
    if isinstance(last, (int, float)) and still_typing:
        return values[:-1]
    return values


POLICIES = {"the SDK's parsed snapshot": the_sdk,
            "every string as it arrives": strings_as_they_arrive,
            "numbers once something follows": numbers_once_ended}

with ThreadPoolExecutor(max_workers=3) as pool:
    streams = list(pool.map(snapshots, REVIEWS))
print(f"{len(REVIEWS)} quarterly reviews extracted as a streamed structured output; "
      f"{sum(map(len, streams))} snapshots\n")
print(f"  {'shown to the reader':<32}{'values shown':>13}{'wrong numbers':>15}{'wrong text':>12}")
histories: dict[str, list[object]] = {}
first_right: dict[str, dict[tuple[int, str], int]] = {}
for name, policy in POLICIES.items():
    shown = wrong_numbers = wrong_text = 0
    first_right[name] = {}
    for n, stream in enumerate(streams):
        final = dict(leaves(json.loads(stream[-1])))
        displayed: set[tuple[str, str]] = set()
        for chunk, text in enumerate(stream):
            for path, value in policy(text):
                if value == final.get(path):
                    first_right[name].setdefault((n, path), chunk)
                key = (path, json.dumps(value))
                if key in displayed:
                    continue
                displayed.add(key)
                shown += 1
                if value != final.get(path):
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        wrong_numbers += 1
                    else:
                        wrong_text += 1
                if name == "the SDK's parsed snapshot" and n == 2 and path == ".revenue_usd":
                    histories.setdefault(path, []).append(value)
    print(f"  {name:<32}{shown:>13}{wrong_numbers:>15}{wrong_text:>12}")

print("\n  wrong = a value put on screen that differs from the one in the finished record")
print(f"\nrevenue_usd for {REVIEWS[2].stem}, as the SDK's snapshots showed it:")
print("  " + " -> ".join(str(v) for v in histories[".revenue_usd"]))

sdk, safe = first_right["the SDK's parsed snapshot"], first_right["numbers once something follows"]
delays = sorted(safe[k] - sdk[k] for k in safe)
print(f"\nWaiting for something to follow a number put each value on screen a median of")
median = statistics.median(delays)
print(f"{median:.0f} chunk{'' if median == 1 else 's'} after the SDK's snapshot first held its final "
      f"value, and at most {delays[-1]},")
print(f"in records of a median {statistics.median(map(len, streams)):.0f} chunks")
