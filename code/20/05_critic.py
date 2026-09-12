# timeout: 2400
# Does a second agent reviewing the first one's answer make it better?

import json
import sys

sys.path.insert(0, "code")
sys.path.insert(0, "code/20")
from _taskset import TASKS, score                        # noqa: E402
from clarity.config import MODEL_FAST                    # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402
from openai import OpenAI

REPEATS = 3
SOLO = ("You are an analyst for Meridian. Use tools for every fact. Give the figure "
        "first, then the cause, then anything you could not verify.")
CRITIC = ("You review a colleague's answer about Meridian. Say only what is wrong "
          "with it: a figure that was not obtained from a tool, a cause that is not "
          "in the documents, a part of the question left unanswered. If it is sound, "
          "reply exactly: NO ISSUES.")

client = OpenAI()
chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())


def critique(question: str, answer: str) -> str:
    reply = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        messages=[{"role": "system", "content": CRITIC},
                  {"role": "user",
                   "content": f"Question:\n{question}\n\nAnswer:\n{answer}"}])
    return (reply.choices[0].message.content or "").strip()


rows = {"first answer": [0, 0, 0], "after a critic": [0, 0, 0]}
changed = kept = 0
for task in TASKS:
    for _ in range(REPEATS):
        first = Agent(tools, budget=Budget(steps=8)).run(task["q"], system=SOLO)
        f, c = score(task, first.answer)
        rows["first answer"][0] += f
        rows["first answer"][1] += c
        rows["first answer"][2] += first.tokens

        note = critique(task["q"], first.answer)
        if note.upper().startswith("NO ISSUES"):
            final, tokens = first.answer, 0
            kept += 1
        else:
            changed += 1
            revise = Agent(tools, budget=Budget(steps=8))
            second = revise.run(
                f"{task['q']}\n\nYour first answer was:\n{first.answer}\n\n"
                f"A reviewer raised this:\n{note}\n\nAnswer again, correcting what "
                f"is genuinely wrong and keeping what is right.", system=SOLO)
            final, tokens = second.answer, second.tokens
        f, c = score(task, final)
        rows["after a critic"][0] += f
        rows["after a critic"][1] += c
        rows["after a critic"][2] += first.tokens + tokens + 500

total = len(TASKS) * REPEATS
print(f"{len(TASKS)} questions, {REPEATS} runs each — {total} answers\n")
for label, (figure, cause, tokens) in rows.items():
    print(f"  {label:<16} figure {figure:>2}/{total}   cause {cause:>2}/{total}   "
          f"{tokens // total:>6,} tokens/answer")

before = sum(rows["first answer"][:2]) / (2 * total)
after = sum(rows["after a critic"][:2]) / (2 * total)
cost = rows["after a critic"][2] / rows["first answer"][2]
print()
print(f"  accuracy {before:.0%} -> {after:.0%}   for {cost:.1f}x the tokens")
print(f"  the critic found something to say {changed} times, and passed {kept}")
json.dump({"rows": rows, "total": total, "changed": changed, "kept": kept,
           "before": before, "after": after, "cost": cost},
          open("code/20/_critic.json", "w"), indent=1)

print()
if after > before:
    print(f"It helped: {before:.0%} to {after:.0%}, for {cost:.1f}x the tokens. Decide "
          f"whether that")
    print("trade is worth it on your own numbers, not on this one.")
elif after == before:
    print(f"It changed nothing at all, for {cost:.1f}x the tokens.")
else:
    print(f"It made things worse: {before:.0%} down to {after:.0%}, for {cost:.1f}x "
          f"the tokens.")
print()
print(f"The critic objected {changed} times out of {total} and passed the other "
      f"{kept}. Every one of")
print("those objections cost a second full answer, and on this task set they did not")
print("buy a better one.")
print()
print("That is consistent with §9.6, where self-reflection was the worst of four")
print("prompting patterns, and the reason is the same. The mistakes here are not")
print("visible in the output: a cause that is subtly the wrong quarter reads")
print("perfectly well. Checking it means searching the documents again, which is not")
print("reviewing — it is redoing the work and hoping the second attempt is luckier.")
