# timeout: 2400
# Cases written by a model, against cases written by hand, from the same six documents. Are the
# model's questions closer to the document's own words, and easier for Clarity to answer? The
# generator is MODEL_SMART; Clarity answers with MODEL_FAST, as in this chapter's first score.

import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

sys.path.insert(0, "code")
from clarity.config import MODEL_SMART                           # noqa: E402
from clarity.evals.runner import correct, load, normalise        # noqa: E402
from clarity.v0_6.retrieve import Retriever                      # noqa: E402
from clarity.v0_7.warehouse import Warehouse                     # noqa: E402
from clarity.v0_8.tools import build_tools                       # noqa: E402
from clarity.v0_9.agent import Agent, Budget                     # noqa: E402
from meridian_index import load_index                            # noqa: E402

client = OpenAI()
DOCS = Path("data/meridian/documents")
SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. If the answer "
          "is not in the documents or the warehouse, say so plainly rather than guessing.")
STOP = set("what which when where does were with from that this have been their there about "
           "under much many".split())


class Case(BaseModel):
    question: str
    answer: str
    quote: str


class Cases(BaseModel):
    cases: list[Case]


def find(name: str) -> Path:
    return next(DOCS.rglob(name))


def generate(name: str) -> list[dict]:
    text = find(name).read_text()
    parsed = client.chat.completions.parse(
        model=MODEL_SMART, max_completion_tokens=4000, response_format=Cases,
        messages=[{"role": "system", "content":
                   "You write evaluation cases for a question-answering system that serves "
                   "Meridian Supply Co.'s analysts. Write five questions an analyst might ask "
                   "that this document answers. For each, give the answer as the fact alone — a "
                   "number, name, date or short phrase, written exactly as the document writes "
                   "it — and the passage from the document, copied verbatim, that contains it."},
                  {"role": "user", "content": f"<document name='{name}'>\n{text}\n</document>"}],
    ).choices[0].message.parsed
    return [{"question": c.question, "answer": c.answer, "span": c.quote, "source": name,
             "kind": "document", "accept": []} for c in (parsed.cases if parsed else [])]


def verified(case: dict) -> bool:
    """§21.4's check: the quote is in the document, and the answer is in the quote."""
    document = normalise(find(case["source"]).read_text())
    return normalise(case["span"]) in document and normalise(case["answer"]) in normalise(case["span"])


def names_its_document(case: dict) -> bool:
    """Does the question say which document it is about, as a question put to the whole corpus must?"""
    q, source = case["question"], case["source"]
    if source.startswith("contract-"):
        return source.removeprefix("contract-").removesuffix(".md") in q
    if source.startswith("qbr-"):
        year, quarter = source.removesuffix(".md").split("-")[1:]
        return year in q and quarter in q
    return any(depot in q for depot in ("Newark", "Atlanta", "Columbus", "Sacramento", "Dallas"))


def overlap(case: dict) -> float:
    """The share of the question's content words that appear in the passage that answers it."""
    words = {w for w in re.findall(r"[a-z]{4,}", case["question"].lower()) if w not in STOP}
    span = case["span"].lower()
    return sum(w in span for w in words) / len(words) if words else 0.0


by_hand = [c for c in load() if c["tier"] == "tail" and c["kind"] != "unanswerable"]
sources = sorted({c["source"] for c in by_hand})
with ThreadPoolExecutor(max_workers=len(sources)) as pool:
    written = [c for batch in pool.map(generate, sources) for c in batch]
kept = [c for c in written if verified(c)]

chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())


def right(case: dict) -> bool:
    reply = Agent(tools, budget=Budget(steps=6)).run(case["question"], system=SYSTEM).answer
    return correct(case, reply)


with ThreadPoolExecutor(max_workers=8) as pool:
    hand_scores = list(pool.map(right, by_hand))
    model_scores = list(pool.map(right, kept))
hand_right, model_right = sum(hand_scores), sum(model_scores)

print(f"{len(sources)} documents; {len(by_hand)} cases written by hand, {len(written)} by a model, "
      f"{len(kept)} verified\n")
print(f"  {'cases':<20}{'words in passage':>18}{'names its document':>20}{'Clarity right':>15}")
for label, cases, scores in (("written by hand", by_hand, hand_scores),
                             ("written by a model", kept, model_scores)):
    named = sum(map(names_its_document, cases))
    print(f"  {label:<20}{statistics.median(map(overlap, cases)):>18.0%}{named:>16}/{len(cases):<3}"
          f"{sum(scores):>11}/{len(cases)}")
named_scores = [s for c, s in zip(kept, model_scores) if names_its_document(c)]
unnamed_scores = [s for c, s in zip(kept, model_scores) if not names_its_document(c)]
print(f"\n  model questions naming their document: {sum(named_scores)}/{len(named_scores)} right;"
      f" not naming it: {sum(unnamed_scores)}/{len(unnamed_scores)}")
print("\n  words in passage = median share of a question's content words in its passage;")
print("  names its document = the contract reference, year and quarter, or depot")
print("\n  the model's questions that did not name their document, for example:")
for case in [c for c in kept if not names_its_document(c)][:3]:
    print(f"    {case['question']}")
json.dump({"written": written, "kept": [c["question"] for c in kept]},
          open("code/21/_model_written.json", "w"), indent=2)
