# Which request parameters each model actually accepts, found by asking rather than
# assuming. Every probe is a tiny request; a rejection costs nothing.

import openai
from openai import OpenAI

from clarity.config import MODEL_FAST, MODEL_SMART

client = OpenAI()
BASE = {"messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_completion_tokens": 400}

PROBES = {
    "temperature=0.2":        {"temperature": 0.2},
    "top_p=0.5":              {"top_p": 0.5},
    "n=2":                    {"n": 2},
    "seed=7":                 {"seed": 7},
    "presence_penalty=0.5":   {"presence_penalty": 0.5},
    "frequency_penalty=0.5":  {"frequency_penalty": 0.5},
    "logprobs + top_logprobs": {"logprobs": True, "top_logprobs": 2},
    "logit_bias":             {"logit_bias": {"15": -100}},
    "stop=['.']":             {"stop": ["."]},
    "max_tokens=20":          {"max_tokens": 20, "max_completion_tokens": None},
    "reasoning_effort=low":   {"reasoning_effort": "low"},
}


def probe(model: str, extra: dict) -> str:
    kwargs = {**BASE, **extra}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        client.chat.completions.create(model=model, **kwargs)
        return "accepted"
    except openai.BadRequestError as error:
        return "rejected"
    except openai.APIError as error:                          # noqa: F841
        return type(error).__name__


results = {name: (probe(MODEL_FAST, extra), probe(MODEL_SMART, extra))
           for name, extra in PROBES.items()}

print(f"  {'parameter':26} {'MODEL_FAST':>12} {'MODEL_SMART':>12}")
for name, (fast, smart) in results.items():
    print(f"  {name:26} {fast:>12} {smart:>12}")

differ = [name for name, (fast, smart) in results.items() if fast != smart]
print(f"\n{len(differ)} of {len(PROBES)} parameters are accepted by one model and not the other.")
