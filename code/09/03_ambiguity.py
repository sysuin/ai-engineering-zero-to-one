# timeout: 900
# A model asked an ambiguous question will answer it. Giving it somewhere else to go.

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contract = load()[0]

# Five questions about one contract. Two are answerable from it, three are not —
# either because the contract is silent, or because the question is ambiguous.
QUESTIONS = [
    ("answerable", "How many days notice is required before a price rise?"),
    ("answerable", "What is the price adjustment cap?"),
    ("not stated", "What is the annual contract value?"),
    ("not stated", "Who is the named account manager?"),
    ("ambiguous",  "What is the notice period?"),
]


class Answer(BaseModel):
    """An answer, or an honest reason there isn't one."""
    status: Literal["answered", "not_in_document", "question_is_ambiguous"] = Field(
        description="answered only if the document states it unambiguously.")
    answer: str | None = Field(
        description="The answer, quoting the document. Null unless status is answered.")
    clarification_needed: str | None = Field(
        description="If the question is ambiguous, what you need the asker to specify.")


BODY = f"<contract>\n{contract['text']}\n</contract>\n\nQuestion: "


def plain(question: str) -> str:
    return (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200,
        messages=[{"role": "user", "content": BODY + question}],
    ).choices[0].message.content or "").strip().replace("\n", " ")


def with_escape_hatch(question: str) -> Answer:
    return client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=300,
        response_format=Answer,
        messages=[
            {"role": "system", "content":
             "Answer only what the document states. If the document does not state it, "
             "say so. If the question could mean more than one thing, say what needs "
             "clarifying instead of choosing one."},
            {"role": "user", "content": BODY + question},
        ],
    ).choices[0].message.parsed


with ThreadPoolExecutor(max_workers=8) as pool:
    plains = list(pool.map(lambda q: plain(q[1]), QUESTIONS))
    hatched = list(pool.map(lambda q: with_escape_hatch(q[1]), QUESTIONS))

for (kind, question), free, structured in zip(QUESTIONS, plains, hatched):
    print(f"[{kind}] {question}")
    print(f"   plain      {free[:96]}")
    print(f"   with hatch status={structured.status}")
    if structured.answer:
        print(f"              answer={structured.answer[:70]!r}")
    if structured.clarification_needed:
        print(f"              needs={structured.clarification_needed[:70]!r}")
    print()

honest = sum(1 for (kind, _), s in zip(QUESTIONS, hatched)
             if (kind == "answerable") == (s.status == "answered"))
print(f"The structured version classified {honest} of {len(QUESTIONS)} correctly.")
print()
print("The contract does not name an account manager and never mentions annual value.")
print("Asked plainly, the model must produce something, because producing something is")
print("what it does. Given a status field with somewhere else to go, it goes there.")
print()
print("'I don't know' has to be a representable value before a model can choose it.")

Path("code/09/_ambiguity.json").write_text(json.dumps(
    [{"kind": k, "question": q, "status": s.status} for (k, q), s in
     zip(QUESTIONS, hatched)], indent=2))
