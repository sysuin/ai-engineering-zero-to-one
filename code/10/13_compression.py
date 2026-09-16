# timeout: 1800
# Three ways to send less of each document, measured. Every quarterly-review question in the
# golden set, asked over its own review and three others; the context is sent whole, cut to the
# sentences a model selects, or rewritten as a summary — for this question, or once per document.

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST
from clarity.evals.runner import abstained, correct, load

client = OpenAI()
FOLDER = Path("data/meridian/documents/quarterly-reviews")
NAMES = sorted(p.name for p in FOLDER.glob("qbr-*.md"))
TEXT = {name: (FOLDER / name).read_text() for name in NAMES}
cases = [c for c in load() if c["kind"] == "document" and c["source"] in TEXT]


def documents_for(case: dict) -> list[str]:
    i = NAMES.index(case["source"])
    return [NAMES[(i + k) % len(NAMES)] for k in (2, 0, 5, 9)]      # the source is second


def ask(prompt: str, tokens: int = 300) -> tuple[str, int]:
    reply = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=tokens,
        messages=[{"role": "user", "content": prompt}])
    return reply.choices[0].message.content or "", reply.usage.prompt_tokens


SELECT = ("Copy, exactly as written, only the lines from these "
          "documents that are needed to answer the question. "
          "Copy nothing else.")
SELECT_LABELLED = ("Copy, exactly as written, only the lines from "
                   "these documents that are needed to answer the "
                   "question, each preceded by the title and period "
                   "of the document it came from. Copy nothing else.")
SUMMARISE = ("Summarise these documents in at most 80 words, keeping "
             "whatever is needed to answer the question.")


with ThreadPoolExecutor(max_workers=12) as pool:
    once = dict(zip(NAMES, pool.map(
        lambda n: ask(f"Summarise this quarterly review in at most 60 words.\n\n{TEXT[n]}")[0],
        NAMES)))


def contexts(case: dict) -> dict[str, tuple[str, int]]:
    """Each kind of context, with the input tokens it took to make it."""
    names = documents_for(case)
    whole = "\n\n".join(TEXT[n] for n in names)
    per_question = {kind: ask(f"{instruction}\n\n{whole}\n\nQuestion: {case['question']}")
                    for kind, instruction in (("selected lines", SELECT),
                                              ("lines with their source", SELECT_LABELLED),
                                              ("summary for the question", SUMMARISE))}
    return {"whole documents": (whole, 0), **per_question,
            "summary made once": ("\n\n".join(f"{n}: {once[n]}" for n in names), 0)}


def answer(context: str, question: str) -> tuple[str, int]:
    return ask("Answer the question using only the context. If the context does not contain the "
               f"answer, say it does not contain it.\n\nContext:\n{context}\n\nQuestion: {question}", 120)


def figures(text: str) -> set[str]:
    return {m.replace(",", "") for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def kind_of(value: float, written: str, source: list[float]) -> int:
    """0: a figure the source contains; 1: a source figure rounded, perhaps to thousands or
    millions; 2: neither."""
    if value in source:
        return 0
    places = len(written.split(".")[1]) if "." in written else 0
    for s in source:
        for scale in (1, 1_000, 1_000_000):
            if round(s / scale, places) == value:
                return 1
    return 2


with ThreadPoolExecutor(max_workers=12) as pool:
    built = list(pool.map(contexts, cases))
    results = {kind: list(pool.map(lambda pair: answer(pair[0][kind][0], pair[1]["question"]),
                                   zip(built, cases)))
               for kind in built[0]}

print(f"{len(cases)} questions, each over four quarterly reviews (its own and three others)\n")
print(f"  {'':<26}{'':>6}{'input tokens':>13}{'figures in the context':>30}")
print(f"  {'context sent':<26}{'right':>6}{'answer':>8}{'all':>7}"
      f"{'copied':>12}{'rounded':>9}{'other':>7}")
for kind, outs in results.items():
    right = sum(correct(c, a) for (a, _), c in zip(outs, cases))
    answer_tokens = sum(t for _, t in outs) / len(outs)
    all_tokens = answer_tokens + sum(ctx[kind][1] for ctx in built) / len(built)
    tally = [0, 0, 0]
    for ctx, c in zip(built, cases):
        source = [float(f) for f in figures("\n".join(TEXT[n] for n in documents_for(c)))]
        for f in figures(ctx[kind][0]):
            tally[kind_of(float(f), f, source)] += 1
    print(f"  {kind:<26}{right:>6}{answer_tokens:>8,.0f}{all_tokens:>7,.0f}"
          f"{tally[0]:>12}{tally[1]:>9}{tally[2]:>7}")
print(f"  (making the once-only summaries took {len(NAMES)} calls, shared by every question)")


def period(case: dict) -> str:
    return case["source"][4:11].replace("-", " ")             # qbr-2024-Q3.md -> 2024 Q3


selections = [(ctx["selected lines"][0], c) for ctx, c in zip(built, cases)]
unlabelled = [(text, c) for text, c in selections if period(c) not in text]
print(f"\n  selections that do not say which quarter they came from: {len(unlabelled)} of {len(cases)}")
text, c = unlabelled[0]
print(f"    {c['question']!r} -> selected: {text.strip()[:60]!r}")
misses = [c for (a, _), c in zip(results["summary made once"], cases) if not correct(c, a)]
refused = sum(abstained(a) for (a, _), c in zip(results["summary made once"], cases)
              if not correct(c, a))
print(f"  misses with the summary made once: {len(misses)}; {refused} said it was not there")
if misses:
    c = misses[0]
    print(f"    {c['question']!r} (answer {c['answer']}); the summary:")
    print(f"    {once[c['source']][:62]!r}")
