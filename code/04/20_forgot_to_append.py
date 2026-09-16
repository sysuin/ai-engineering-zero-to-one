# timeout: 600
# What happens if the assistant's reply is not appended before the next turn? Twenty two-turn
# conversations in which the second question depends on the first answer, with the reply kept
# and with it forgotten.

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
FIRST = f"Pick one of these regions at random and reply with its name only: {', '.join(REGIONS)}."
SECOND = "Repeat the region you just gave me, name only."


def named(text: str) -> str | None:
    found = [r for r in REGIONS if r.lower() in text.lower()]
    return max(found, key=len) if found else None       # "Southwest" contains "West"


picks = Counter()


def conversation(keep_reply: bool) -> str:
    messages = [{"role": "user", "content": FIRST}]
    first = client.chat.completions.create(model=MODEL_FAST, messages=messages)
    picked = named(first.choices[0].message.content or "")
    picks[picked] += 1
    if keep_reply:
        messages.append({"role": "assistant", "content": first.choices[0].message.content})
    messages.append({"role": "user", "content": SECOND})
    second = client.chat.completions.create(model=MODEL_FAST, messages=messages)
    said = named(second.choices[0].message.content or "")
    if said is None:
        return "said it did not know"
    return "same region" if said == picked else "a different region"


OUTCOMES = ("same region", "a different region", "said it did not know")
print(f"  {'the reply was':<12}" + "".join(f"{o:>22}" for o in OUTCOMES))
for keep in (True, False):
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda _: conversation(keep), range(20)))
    print(f"  {'kept' if keep else 'forgotten':<12}" + "".join(f"{results.count(o):>19}/20"
                                                            for o in OUTCOMES))
print("\n  the 'random' first picks, across all forty: "
      + ", ".join(f"{region} {n}" for region, n in picks.most_common()))
