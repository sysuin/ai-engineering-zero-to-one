# timeout: 600
# Provider-side prompt caching reuses the longest identical prefix of a request. The same
# long instructions, the same questions — once with the stable part first, once with a
# timestamp in front of it.

import time

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
INSTRUCTIONS = ("You are an analyst for Meridian Supply Co. " +
                "Answer only from the figures provided, state the figure first, and say plainly "
                "when a figure is missing. " * 150)         # long enough for the cache to apply
QUESTIONS = ["Which quarter had the highest revenue?", "What was the lowest margin?",
             "How many quarters are listed?", "Which quarter grew fastest?"]
FIGURES = "2024 Q3 revenue 8,461,842; 2024 Q4 8,266,068; 2025 Q1 9,079,542; 2025 Q2 8,633,631."


def ask(system: str, question: str) -> tuple[int, int]:
    usage = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=60,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"{FIGURES}\n\n{question}"}]).usage
    cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
    return usage.prompt_tokens, cached


for label, make_system in (
        ("stable instructions first", lambda: INSTRUCTIONS + "\n\nRequest time: " + time.strftime("%H:%M:%S")),
        ("timestamp first", lambda: "Request time: " + time.strftime("%H:%M:%S") + "\n\n" + INSTRUCTIONS)):
    print(label)
    total = cached_total = 0
    for question in QUESTIONS:
        prompt, cached = ask(make_system(), question)
        total, cached_total = total + prompt, cached_total + cached
        print(f"  {prompt:>5} prompt tokens, {cached:>5} of them cached")
        time.sleep(1.1)                                     # so every timestamp differs
    print(f"  {cached_total / total:.0%} of all prompt tokens were served from the cache\n")
