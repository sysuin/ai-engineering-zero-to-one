# timeout: 900
# Follow-up questions do not stand alone. Retrieval sees only the words it is given, so
# "And how full is it?" finds nothing about the depot the conversation was discussing.

import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST       # noqa: E402
from clarity.v0_5.rag import Clarity         # noqa: E402
from meridian_index import load_index        # noqa: E402

chunks, vectors = load_index()
clarity = Clarity(chunks, vectors)
client = OpenAI()

# (previous question, follow-up, a phrase the right chunk must contain)
CONVERSATIONS = [
    ("Why did Halloway Group not renew?", "Which region was that account in?", "Halloway"),
    ("What is the racked storage at the Dallas depot?", "And how full was it at year end?", "Utilisation"),
    ("Tell me about the goods received note from Columbus.", "Which line was short?", "12 short"),
    ("What does the specification appendix say about item 118?", "What revision is it at?", "Item 118"),
    ("What did the December 2025 newsletter say about revenue?", "And what was the notice about the depot?", "inventory count"),
    ("What notice is needed to terminate a supply agreement?", "And before a price rise?", "price"),
]


def condense(previous: str, follow_up: str) -> str:
    return (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=60,
        messages=[{"role": "user", "content":
                   "Rewrite the follow-up as a single standalone question that makes sense "
                   "without the conversation. Reply with the question only.\n\n"
                   f"Previous question: {previous}\nFollow-up: {follow_up}"}],
    ).choices[0].message.content or "").strip()


def found(question: str, phrase: str) -> bool:
    return any(phrase.lower() in p.text.lower() for p in clarity.retrieve(question, k=6))


with ThreadPoolExecutor(max_workers=6) as pool:
    rewritten = list(pool.map(lambda c: condense(c[0], c[1]), CONVERSATIONS))

totals = {"follow-up alone": 0, "previous + follow-up": 0, "condensed": 0}
print(f"  {'follow-up':42} {'alone':>6} {'joined':>7} {'condensed':>10}")
for (previous, follow_up, phrase), standalone in zip(CONVERSATIONS, rewritten):
    marks = [found(follow_up, phrase), found(f"{previous} {follow_up}", phrase),
             found(standalone, phrase)]
    for key, ok in zip(totals, marks):
        totals[key] += ok
    print(f"  {follow_up[:42]:42} " + " ".join(f"{'yes' if m else 'no':>{w}}"
                                              for m, w in zip(marks, (6, 7, 10))))
print("\n  right chunk in the top 6: " + ", ".join(f"{k} {v}/{len(CONVERSATIONS)}"
                                                 for k, v in totals.items()))
print(f"\n  e.g. {CONVERSATIONS[1][1]!r} -> {rewritten[1]!r}")
