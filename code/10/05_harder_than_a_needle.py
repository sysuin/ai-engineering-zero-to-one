# timeout: 1800
# A single planted fact is the easiest thing to find in a long document. Four harder
# questions over the same 400-section appendix, three runs each — and, for the questions
# about the whole document, the alternative of extracting every section and counting in code.

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tiktoken
from openai import OpenAI
from pydantic import BaseModel

from clarity.config import MODEL_FAST

client = OpenAI()
encoder = tiktoken.encoding_for_model(MODEL_FAST)
APPENDIX = Path("data/meridian/documents/awkward/03-long-appendix.md").read_text()
SECTIONS = ["## " + s for s in re.split(r"^## ", APPENDIX, flags=re.M)[1:]]
RUNS = 3


def with_inserts(inserts: dict[float, str]) -> str:
    body = list(SECTIONS)
    for depth in sorted(inserts, reverse=True):
        body.insert(int(len(body) * depth), inserts[depth])
    return "# Appendix C — Product specifications\n\n" + "\n".join(body)


def ask(document: str, question: str) -> str:
    return (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        messages=[{"role": "user", "content": f"<document>\n{document}\n</document>\n\n{question}"}],
    ).choices[0].message.content or "").strip()


not_approved = sorted(int(re.search(r"item (\d+)", s).group(1)) for s in SECTIONS
                      if "not approved for food-contact" in s)

TESTS = [
    ("combine two facts",
     with_inserts({0.15: "## C.990 Columbus depot\n\nThe Columbus depot holds 1,240 pallets.\n",
                   0.85: "## C.991 Dayton depot\n\nThe Dayton depot holds 875 pallets.\n"}),
     "How many pallets do the Columbus and Dayton depots hold in total? Reply with the number only.",
     lambda a: "2115" in a.replace(",", "")),
    ("answer is absent",
     with_inserts({0.5: "## C.990 Columbus depot\n\nThe Columbus override code is QX-4471.\n"}),
     "What is the Toledo depot override code? If the document does not say, reply NOT STATED.",
     lambda a: "NOT STATED" in a.upper()),
    ("a later fact supersedes",
     with_inserts({0.2: "## C.990 Columbus depot\n\nThe Columbus override code is QX-4471.\n",
                   0.8: "## C.991 Notice of change\n\nWith effect from March, the Columbus "
                        "override code QX-4471 is withdrawn and replaced by QX-5190.\n"}),
     "What is the current Columbus override code? Reply with the code only.",
     lambda a: "QX-5190" in a and "QX-4471" not in a),
    ("count across the document",
     with_inserts({}),
     "How many specification items are NOT approved for food-contact use? Reply with the number only.",
     lambda a: str(len(not_approved)) == re.sub(r"\D", "", a)),
]

print(f"appendix: {len(SECTIONS)} sections, {len(encoder.encode(APPENDIX)):,} tokens\n")
print(f"  {'question':28} {'right':>7}   a sample answer")
for label, document, question, check in TESTS:
    with ThreadPoolExecutor(max_workers=RUNS) as pool:
        answers = list(pool.map(lambda _: ask(document, question), range(RUNS)))
    right = sum(check(a) for a in answers)
    print(f"  {label:28} {right}/{RUNS}     {answers[0][:40]!r}")

print(f"\nthe count, done the other way: extract each of the {len(SECTIONS)} sections, count in code")


class Item(BaseModel):
    item: int
    approved_for_food_contact: bool


def extract(section: str) -> Item:
    return client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=60, response_format=Item,
        messages=[{"role": "user", "content": section}],
    ).choices[0].message.parsed


with ThreadPoolExecutor(max_workers=16) as pool:
    items = list(pool.map(extract, SECTIONS))
found = sorted(i.item for i in items if not i.approved_for_food_contact)
print(f"  counted {len(found)}; the true number is {len(not_approved)}; "
      f"same items: {found == not_approved}")
