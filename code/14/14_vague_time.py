# timeout: 600
# "A model asked to infer facets will decide that 'recent performance' means 2025." Twelve
# questions that name no period, sent to Clarity v0.6's facet extractor with its instruction —
# extract only what is named, do not infer — and with an instruction that invites inference.

from concurrent.futures import ThreadPoolExecutor

import sys
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                 # noqa: E402
from clarity.v0_6.retrieve import Facets              # noqa: E402

client = OpenAI()
RUNS = 2
QUESTIONS = ["How has recent performance looked?", "What happened last quarter?",
             "How did the Midwest do this year?", "Any delivery problems lately?",
             "What was revenue in the most recent quarter?",
             "Compare the latest quarter with the one before it.", "How did we finish last year?",
             "What is the current gross margin?", "Summarise performance so far this year.",
             "Did revenue fall recently?", "What changed over the past few months?",
             "What was revenue at the start of the year?"]
PROMPTS = {"only what is named (v0.6)": "Extract only what the question explicitly names. "
                                        "Null otherwise. Do not infer.",
           "what it refers to": "Extract the year and quarter the question refers to."}


def facets(system: str, question: str) -> Facets:
    return client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200, response_format=Facets,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
    ).choices[0].message.parsed


print(f"{len(QUESTIONS)} questions that name no year or quarter, {RUNS} runs each\n")
print(f"  {'instruction':<30}{'invented a year':>17}{'or a quarter':>14}{'years given':>14}")
for name, system in PROMPTS.items():
    jobs = [q for q in QUESTIONS for _ in range(RUNS)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        found = list(pool.map(lambda q: facets(system, q), jobs))
    years = sum(f.year is not None for f in found)
    quarters = sum(f.quarter is not None for f in found)
    values = sorted({f.year for f in found if f.year is not None})
    print(f"  {name:<30}{years:>13}/{len(jobs)}{quarters:>10}/{len(jobs)}"
          f"{', '.join(map(str, values)) or '-':>14}")
