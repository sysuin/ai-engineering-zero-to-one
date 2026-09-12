# timeout: 2400
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# The answer is wrong. Which stage is wrong?

import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                          # noqa: E402
from clarity.evals.runner import correct, load, plain          # noqa: E402
from clarity.v0_5.rag import Clarity                           # noqa: E402
from meridian_index import load_index                          # noqa: E402
from openai import OpenAI                                      # noqa: E402

K = 6
client = OpenAI()
chunks, vectors = load_index()
system = Clarity(chunks, vectors)
cases = [c for c in load()
         if c["kind"] == "document" and c.get("span") and c.get("tier") == "tail"]

ANSWERER = ("Answer the question using only the passages given. Be brief. If the "
            "passages do not contain the answer, say so.")


def generate(question: str, passages: list[str]) -> str:
    context = "\n\n".join(passages)
    reply = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=200,
        messages=[{"role": "system", "content": ANSWERER},
                  {"role": "user",
                   "content": f"Passages:\n{context[:6000]}\n\nQuestion: {question}"}])
    return (reply.choices[0].message.content or "").strip()


def bisect(case: dict) -> dict:
    """
    Four questions, in order, each one cheap enough to ask about every failure.

    The order matters: each stage assumes the one before it worked, so the first
    'no' is the answer and everything after it is noise.
    """
    passages = system.retrieve(case["question"], k=K)
    texts = [p.text for p in passages]
    probe = plain(" ".join(case["span"].split())[:60])

    # 1. was the answer retrievable at all?
    in_corpus = any(probe in plain(" ".join(c["text"].split()))
                    for c in chunks)
    # 2. did retrieval fetch it?
    retrieved = any(probe in plain(" ".join(t.split())) for t in texts)
    # 3. given only the right passage, can generation answer?
    with_gold = generate(case["question"], [case["span"]])
    # 4. given what retrieval actually returned, can it?
    with_real = generate(case["question"], texts)

    # A span can be correct evidence and insufficient context: one row of a table,
    # without its headers, is exactly where the fact is and not enough to read it.
    # Calling that "generation" would blame the wrong stage.
    stage = ("the corpus" if not in_corpus else
             "retrieval" if not retrieved else
             "the span alone" if not correct(case, with_gold) else
             "the surrounding context" if not correct(case, with_real)
             else "nothing — it passed")
    return {"id": case["id"], "question": case["question"],
            "in_corpus": in_corpus, "retrieved": retrieved,
            "gold_ok": correct(case, with_gold), "real_ok": correct(case, with_real),
            "stage": stage}


with ThreadPoolExecutor(max_workers=6) as pool:
    rows = list(pool.map(bisect, cases))
json.dump(rows, open("code/24/_bisect.json", "w"), indent=1)

print(f"{len(rows)} questions the documents alone must answer, bisected stage by "
      f"stage.\n")
print(f"  {'':<34}{'in corpus':>10}{'retrieved':>11}{'gold ctx':>10}"
      f"{'real ctx':>10}")
for row in rows:
    print(f"  {row['question'][:32]:<34}"
          f"{'yes' if row['in_corpus'] else 'NO':>10}"
          f"{'yes' if row['retrieved'] else 'NO':>11}"
          f"{'ok' if row['gold_ok'] else 'WRONG':>10}"
          f"{'ok' if row['real_ok'] else 'WRONG':>10}")

blame: dict[str, int] = {}
for row in rows:
    blame[row["stage"]] = blame.get(row["stage"], 0) + 1
print(f"\n  where the failures actually are:\n")
for stage, count in sorted(blame.items(), key=lambda kv: -kv[1]):
    print(f"    {stage:<24}{count:>3} of {len(rows)}")

print()
print("Four questions, asked in that order, and the first 'no' is the diagnosis.")
print()
print("  is it in the corpus?      if not, no amount of prompt work helps and the")
print("                            fix is an ingestion ticket")
print("  did retrieval fetch it?   if not, generation is innocent — it answered")
print("                            correctly about what it was shown")
print("  given the cited span,     if not, the span is evidence rather than context.")
print("  can it answer?            One row of a table is exactly where the fact is")
print("                            and not enough to read it without the headers")
print("  given the real context,   if this is the only 'no', something in the other")
print("  can it answer?            passages is distracting it — the most interesting")
print("                            failure of the four, and the one people never")
print("                            look for")
print()
print("Three of those four checks need no model at all, and the fourth is one call.")
print("The expensive habit is arguing about whether the model is bad at contracts")
print("before establishing whether the contract was in the context.")
print()
print("Note the rows where the cited span fails and the full context succeeds. Those")
print("are not generation bugs and they are not retrieval bugs — they are a golden")
print("set recording the narrowest evidence for an answer rather than enough of the")
print("document to act on it. Chapter 21's spans were built to verify cases, and")
print("this is where that choice shows through.")
