# timeout: 1800
# Does asking for the working still help when the model can reason before it answers? The
# direct prompt and the chain-of-thought prompt, at three settings of reasoning effort.

import re
import statistics
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()
POLICY = ("Meridian's procurement policy requires all three of the following:\n"
          "  1. the price adjustment cap is 8% or less\n"
          "  2. the written notice before a price rise is 60 days or more\n"
          "  3. the payment term is 45 days or less")
PROMPTS = {"direct": "Is this contract compliant? Reply with YES or NO only.",
           "chain of thought": "Think step by step, checking each of the three requirements in turn. "
                               "Then end with a final line: 'ANSWER: YES' or 'ANSWER: NO'."}
EFFORTS = ("none", "low", "medium")


def truth(c) -> bool:
    return c["cap_pct"] <= 8 and c["notice_days"] >= 60 and c["payment_days"] <= 45


def verdict(text: str) -> bool | None:
    answers = re.findall(r"answer:\s*\**\s*(yes|no)\b", text, re.IGNORECASE)
    if answers:
        return answers[-1].lower() == "yes"
    first = re.match(r"\W*(yes|no)\b", text, re.IGNORECASE)
    return first.group(1).lower() == "yes" if first else None


def ask(c, prompt: str, effort: str) -> tuple[bool, int, int, str]:
    response = client.chat.completions.create(
        model=MODEL_FAST, reasoning_effort=effort, max_completion_tokens=4_000,
        messages=[{"role": "user", "content": f"{POLICY}\n\n<contract>\n{c['text']}\n</contract>"
                   f"\n\n{PROMPTS[prompt]}"}])
    usage = response.usage
    reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
    return (verdict(response.choices[0].message.content or "") == truth(c),
            usage.completion_tokens - reasoning, reasoning, response.choices[0].finish_reason)


print(f"{len(contracts)} contracts\n")
print(f"  {'effort':8}{'prompt':18}{'right':>7}{'visible out':>13}{'reasoning':>11}")
for effort in EFFORTS:
    for prompt in PROMPTS:
        with ThreadPoolExecutor(max_workers=12) as pool:
            rows = list(pool.map(lambda c: ask(c, prompt, effort), contracts))
        stopped = sum(r[3] != "stop" for r in rows)
        print(f"  {effort:8}{prompt:18}{sum(r[0] for r in rows):>4}/{len(rows)}"
              f"{statistics.mean(r[1] for r in rows):>13,.0f}{statistics.mean(r[2] for r in rows):>11,.0f}"
              + (f"   ({stopped} cut off)" if stopped else ""))
print("\nvisible out and reasoning are mean tokens per contract; both are billed as output")
