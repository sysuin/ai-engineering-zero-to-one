# timeout: 1200
# Clarity's answer prompt tells the model not to use general knowledge. What does that sentence
# do? Ten contracts, five questions each, asked with the clause that answers removed from the
# excerpts — so any term the answer states came from somewhere other than this contract.

import json
import re
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                  # noqa: E402
from clarity.evals.runner import plain                 # noqa: E402
from clarity.prompts import load as load_prompt        # noqa: E402

client = OpenAI()
CONTRACTS = sorted(Path("data/meridian/documents/contracts").glob("*.md"))[::4]
QUESTIONS = {                                    # the numbered clause that answers, and the question
    "2.": "How much notice must the Supplier give before a price rise under {ref}?",
    "3.": "Within how many days must a valid invoice be paid under {ref}?",
    "5.": "Within how many days of delivery may the Buyer reject goods under {ref}?",
    "6.": "What is the cap on the Supplier's aggregate liability under {ref}?",
    "7.": "How much notice is needed to terminate {ref} for convenience?",
}
PROMPT = load_prompt("answer")
GENERAL = ("Do not use general knowledge about supply chains,\ndistribution or accounting, "
           "however confident you are.\n")
CONSTRAINTS = PROMPT[PROMPT.index("# Constraints"):PROMPT.index("# Format")]
assert GENERAL in PROMPT and CONSTRAINTS
PROMPTS = {"as written": PROMPT,
           "without the general-knowledge line": PROMPT.replace(GENERAL, ""),
           "without any constraints": PROMPT.replace(CONSTRAINTS, ""),
           "no grounding at all": "You answer questions for Meridian's analytics team."}
TERM = re.compile(r"\b\d+\s*(?:days?|%|per ?cent|months?)", re.I)
DENIAL = re.compile(r"\b(?:do|does|did) not (?:state|mention|include|specify|contain|say|give|set)"
                    r"|\bnot (?:stated|specified|mentioned|included|given)|\bno (?:mention|clause)", re.I)


def sections(path: Path) -> dict[str, str]:
    """The contract split at its numbered headings; the preamble keeps the reference."""
    parts = re.split(r"^## ", path.read_text(), flags=re.M)
    found = {"preamble": parts[0]}
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        found[heading.strip()] = body.strip()
    return found


def ask(prompt: str, excerpts: dict[str, str], source: str, question: str) -> str:
    context = "\n\n".join(f"[{source}] {h}\n{t}" for h, t in excerpts.items())
    reply = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=500,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": f"<excerpts>\n{context}\n</excerpts>\n\n"
                                              f"Question: {question}"}])
    return reply.choices[0].message.content or ""


jobs = []
for path in CONTRACTS:
    parts = sections(path)
    ref = re.search(r"MSC-\d{4}-\d{3}", parts["preamble"]).group(0)
    for number, question in QUESTIONS.items():
        heading = next(h for h in parts if h.startswith(number))
        answer_terms = set(TERM.findall(parts[heading]))
        kept = {h: t for h, t in parts.items() if h != heading}
        jobs.append((path.name, question.format(ref=ref), parts, kept, answer_terms))


def classify(answer: str, kept: dict[str, str]) -> str:
    first = re.split(r"(?<=[.!?])\s+", answer.strip(), maxsplit=1)[0]
    stated = TERM.findall(first)
    if not stated or DENIAL.search(first):
        return "said it is missing"
    present = " ".join(kept.values())
    return "a term from another clause" if all(s in present for s in stated) else \
        "a term in no excerpt"


print(f"{len(CONTRACTS)} contracts x {len(QUESTIONS)} questions = {len(jobs)} questions; the clause "
      "that answers each\nis removed from its excerpts\n")
with ThreadPoolExecutor(max_workers=10) as pool:
    control = list(pool.map(lambda j: ask(PROMPT, j[2], j[0], j[1]), jobs))
right = sum(bool(terms & set(TERM.findall(a))) for (*_, terms), a in zip(jobs, control))
print(f"with the clause present, as a check: the right term in {right} of {len(jobs)} answers\n")

KINDS = ("said it is missing", "a term from another clause", "a term in no excerpt")
print(f"  {'prompt':<36}{'missing':>9}{'other clause':>14}{'no excerpt':>12}")
report, examples, saved = {}, {}, []
for name, prompt in PROMPTS.items():
    with ThreadPoolExecutor(max_workers=10) as pool:
        answers = list(pool.map(lambda j: ask(prompt, j[3], j[0], j[1]), jobs))
    kinds = [classify(a, j[3]) for j, a in zip(jobs, answers)]
    counts = {k: kinds.count(k) for k in KINDS}
    print(f"  {name:<36}{counts[KINDS[0]]:>9}{counts[KINDS[1]]:>14}{counts[KINDS[2]]:>12}")
    report[name] = counts
    saved += [{"prompt": name, "question": j[1], "kind": k, "answer": a}
              for j, a, k in zip(jobs, answers, kinds)]
    for job, answer, kind in zip(jobs, answers, kinds):
        if kind != KINDS[0]:
            examples.setdefault(kind, (name, job[1], answer))

print("\n  the answer's first sentence said the term is missing, stated a term found in")
print("  another clause of the contract, or stated one found nowhere")

# 2. Questions the excerpts cannot fully answer by design: whether a term is usual.
NORMS = {"3.": "Is the payment term under {ref} long or short compared with what is usual?",
         "6.": "Is the liability cap under {ref} high or low compared with standard practice?",
         "5.": "Is the rejection window under {ref} generous compared with the norm?"}
NORM = re.compile(r"\b(?:typical|typically|usual|usually|standard|common|commonly|market|"
                  r"generally|normal|normally|customary|industry)\b", re.I)
NEGATION = re.compile(r"\b(?:not|no|cannot|without|missing|benchmark)\b|n't\b", re.I)
CARVE_OUT = re.compile(r"standard (?:carve-outs?|exclusions?)", re.I)     # a phrase, not a judgement


def norm_sentences(answer: str) -> list[str]:
    """Sentences that use a norm word and do not deny knowing the norm."""
    return [s for s in re.split(r"(?<=[.!?])\s+", answer)
            if NORM.search(CARVE_OUT.sub("", s)) and not NEGATION.search(plain(s))]


norm_jobs = []
for path in CONTRACTS:
    parts = sections(path)
    ref = re.search(r"MSC-\d{4}-\d{3}", parts["preamble"]).group(0)
    norm_jobs += [(path.name, q.format(ref=ref), parts) for q in NORMS.values()]
print(f"\n{len(norm_jobs)} questions about whether a term is usual, with every clause present")
print(f"  {'prompt':<36}{'appealed to a norm':>20}")
for name, prompt in PROMPTS.items():
    with ThreadPoolExecutor(max_workers=10) as pool:
        answers = list(pool.map(lambda j: ask(prompt, j[2], j[0], j[1]), norm_jobs))
    flagged = [bool(norm_sentences(a)) for a in answers]
    print(f"  {name:<36}{sum(flagged):>16}/{len(norm_jobs)}")
    report[f"norm: {name}"] = sum(flagged)
    saved += [{"prompt": name, "question": j[1], "kind": "norm" if f else "no norm",
               "answer": a} for j, a, f in zip(norm_jobs, answers, flagged)]
    if name == "without any constraints":
        shown = [(j[1], norm_sentences(a)[0]) for j, a, f in zip(norm_jobs, answers, flagged) if f]
print("\n  appealed to a norm = a sentence calls a term typical, usual or standard, and")
print("  does not say the excerpts cannot tell. Three, without any constraints:")
for question, sentence in shown[:3]:
    print("    " + "\n      ".join(textwrap.wrap(" ".join(sentence.split()), 72)))
json.dump({"report": report, "answers": saved}, open("code/13/_grounding.json", "w"), indent=2)
