# timeout: 1800
# "Deadlock by politeness", measured. Two specialists who may take a question or hand it to the
# other. On questions that clearly belong to one of them and on ones that could belong to either,
# how often does a question bounce, and does the wording of the handover rule change it?

import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from openai import OpenAI                                          # noqa: E402

from clarity.config import MODEL_FAST                              # noqa: E402

client = OpenAI()
RUNS, CAP = 8, 6
ROLES = {"analyst": "You are the analyst. You query Meridian's sales warehouse: revenue, orders, "
                    "margin and units by region, category and quarter.",
         "researcher": "You are the researcher. You read Meridian's documents: quarterly reviews, "
                       "contracts and support tickets, and what they give as reasons."}
RULES = {"polite": "If your colleague may be better placed to handle this, hand it over.",
         "owner": "Take the question unless your colleague's material is required and yours is not "
                  "used at all. Handing over is a cost: a question may be handed over at most once, "
                  "and the note must say what you could not do."}
QUESTIONS = {
    "clear": ["What was total revenue in 2024 Q3?",
              "What does contract MSC-2022-104 say about termination for convenience?"],
    "either": ["Why did the Midwest's numbers look weak in 2024?",
               "Is the margin problem in 2025 a pricing issue or a customer issue?"],
}
TOOLS = [
    {"type": "function", "function": {"name": "take_it", "description": "Take ownership and plan the answer.",
     "parameters": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}}},
    {"type": "function", "function": {"name": "hand_over", "description": "Give the question to your colleague.",
     "parameters": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}}},
]


def run(question: str, rule: str, first: str) -> int:
    """Handovers before someone takes the question; CAP means nobody did."""
    holder, notes = first, []
    for handovers in range(CAP):
        other = "researcher" if holder == "analyst" else "analyst"
        history = "\n".join(notes) or "(none)"
        system = f"{ROLES[holder]} Your colleague: {ROLES[other]} {RULES[rule]}"
        user = f"Question: {question}\nHandover notes so far: {history}"
        call = client.chat.completions.create(
            model=MODEL_FAST, tool_choice="required", tools=TOOLS,
            max_completion_tokens=200,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        ).choices[0].message.tool_calls[0]
        if call.function.name == "take_it":
            return handovers
        note = json.loads(call.function.arguments).get("note", "")
        notes.append(f"{holder}: {note}")
        holder = other
    return CAP


print(f"two specialists, {RUNS} runs per question and starting holder; stop after {CAP} handovers\n")
print(f"  {'rule':8}{'questions':10}{'taken at once':>15}{'bounced 2+':>12}{'never taken':>13}{'mean handovers':>16}")
for rule in RULES:
    for kind, questions in QUESTIONS.items():
        jobs = [(q, rule, first) for q in questions for first in ROLES for _ in range(RUNS)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            counts = list(pool.map(lambda j: run(*j), jobs))
        n = len(counts)
        print(f"  {rule:8}{kind:10}{sum(c == 0 for c in counts) / n:>15.0%}"
              f"{sum(c >= 2 for c in counts) / n:>12.0%}{sum(c == CAP for c in counts) / n:>13.0%}"
              f"{sum(counts) / n:>16.2f}")
