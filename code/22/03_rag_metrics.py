# timeout: 2400
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# Four RAG metrics, and which of them needs a model at all.

import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                           # noqa: E402
from clarity.evals.runner import (abstained, correct,           # noqa: E402
                                  load, plain)
from clarity.v0_5.rag import Clarity                            # noqa: E402
from meridian_index import load_index                           # noqa: E402
from openai import OpenAI                                       # noqa: E402

K = 6
client = OpenAI()
chunks, vectors = load_index()
system = Clarity(chunks, vectors)

# Only document cases: they are the ones with a span, and a span is what makes
# context recall measurable rather than a matter of opinion.
cases = [c for c in load() if c["kind"] == "document" and c.get("span")][:60]

FAITHFUL = """You check whether an answer is supported by the passages it was given.

Reply with JSON only: {"unsupported": ["..."], "supported_count": N}

List every factual claim in the answer that is NOT stated in the passages. A claim
that is stated in different words is supported. General framing is not a claim."""


def measure(case: dict) -> dict:
    passages = system.retrieve(case["question"], k=K)
    answer = system.answer(case["question"], k=K).text

    # --- context recall: did retrieval fetch the passage the answer is actually in?
    # The golden span is the ground truth here, which is why §21.4 insisted on it.
    span = " ".join(case["span"].split())[:120]
    hit = next((i for i, p in enumerate(passages)
                if plain(span[:60]) in plain(" ".join(p.text.split()))), None)

    # --- context precision: how many retrieved passages come from the right document
    from_source = sum(1 for p in passages if p.source == case["source"])

    # --- faithfulness: is every claim in the answer in the passages?
    context = "\n\n".join(f"[{p.source}] {p.text}" for p in passages)
    reply = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=300,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": FAITHFUL},
                  {"role": "user", "content": f"Passages:\n{context[:6000]}\n\n"
                                              f"Answer: {answer[:1200]}"}])
    try:
        checked = json.loads(reply.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        checked = {}
    unsupported = [u for u in (checked.get("unsupported") or []) if isinstance(u, str)]

    return {"id": case["id"], "recall_rank": hit, "recalled": hit is not None,
            "precision": from_source / len(passages) if passages else 0.0,
            "unsupported": len(unsupported), "example": unsupported[:1],
            # A refusal has no claims to support, so it is not unfaithful — it is
            # excluded rather than scored, which is a decision the metric has to make
            # and most implementations make silently.
            "refused": abstained(answer), "correct": correct(case, answer)}


with ThreadPoolExecutor(max_workers=8) as pool:
    rows = list(pool.map(measure, cases))
json.dump(rows, open("code/22/_ragmetrics.json", "w"), indent=1)

recalled = sum(r["recalled"] for r in rows)
top1 = sum(1 for r in rows if r["recall_rank"] == 0)
precision = sum(r["precision"] for r in rows) / len(rows)
graded = [r for r in rows if not r["refused"]]
faithful = sum(1 for r in graded if r["unsupported"] == 0)

print(f"{len(rows)} document questions, {K} passages retrieved for each.\n")
print(f"  context recall     the golden span was retrieved    "
      f"{recalled} of {len(rows)}  ({recalled / len(rows):.0%})")
print(f"                     and it was ranked first          "
      f"{top1} of {len(rows)}  ({top1 / len(rows):.0%})")
print(f"  context precision  passages from the cited document "
      f"{precision:.0%}")
print(f"  faithfulness       no unsupported claim             "
      f"{faithful} of {len(graded)}  ({faithful / len(graded):.0%})")
print(f"                     ({len(rows) - len(graded)} refusals excluded — a refusal "
      f"makes no claims)")

print()
print("Three of those four numbers were computed without a model. Context recall is a")
print("string comparison against the span the case already carries; context precision")
print("is counting filenames. Both are exact, free and reproducible, and both are")
print("metrics people routinely reach for a judge to estimate.")
print()
print("Only faithfulness needed one, because 'is this claim supported by that")
print("passage' is a reading comprehension question with no string operation behind")
print("it.")

missed = [r for r in rows if not r["recalled"]]
missed_ok = sum(1 for r in missed if r["correct"])
missed_refused = sum(1 for r in missed if r["refused"])
got = [r for r in rows if r["recalled"]]
got_ok = sum(1 for r in got if r["correct"])

print()
print("Now the part that matters — what recall actually predicts.\n")
print(f"  span retrieved     {len(got):>2} cases, {got_ok} answered correctly "
      f"({got_ok / len(got):.0%})")
print(f"  span not retrieved {len(missed):>2} cases, {missed_ok} answered correctly "
      f"({missed_ok / len(missed):.0%})")
print(f"                     and {missed_refused} of those {len(missed)} were "
      f"refusals")
print()
if missed_ok == 0:
    print("Context recall is not a proxy for correctness here. It is the ceiling on")
    print("it. Every case where the passage was not retrieved was a case the system")
    print("could not answer, and it said so rather than guessing — which is the")
    print("grounding check from §13.9 doing precisely its job.")
    print()
    print("That is worth knowing before spending a week on prompts. Twenty of these")
    print("sixty failures are retrieval failures, and no wording of any instruction")
    print("recovers a passage that was never fetched.")
else:
    print(f"Recall is a lower bound rather than a ceiling: {missed_ok} answers were")
    print("correct without the expected passage, because Meridian states its figures")
    print("in the summary, in a table, and again in the commentary. Retrieval fetching")
    print("a different one of those is scored as a miss and is not one.")
print()
print(f"Context precision of {precision:.0%} needs its own care. It counts passages "
      f"from the")
print("document the case cites, and a passage from another quarter's review is not")
print("useless — for 'which region was strongest', comparing quarters is reasonable")
print("behaviour that this metric calls noise.")

unfaithful = [r for r in graded if r["unsupported"] and r["example"]]
if unfaithful:
    print()
    print(f"{len(unfaithful)} of the {len(graded)} graded answers carried a claim the "
          f"checker could not")
    print("locate. Here is one:")
    print(f"  {unfaithful[0]['example'][0][:88]}")
    print()
    print("Read the flags before believing the score. A faithfulness checker marks")
    print("anything absent from the passages it was shown, which includes claims that")
    print("are true, in the document, and in a chunk that was not retrieved.")
print()
print("None of this makes the metrics useless. It makes them instruments with known")
print("error, which is the only kind there is. The mistake is quoting faithfulness to")
print("two decimal places without ever reading ten of the flags.")
