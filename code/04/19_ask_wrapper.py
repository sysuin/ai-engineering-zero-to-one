# The six lines from the start of the chapter, grown into the function every later
# chapter wishes it had from day one: it records what happened and refuses to hide a
# truncated answer.

import logging
import sys
import time
from dataclasses import dataclass

from openai import OpenAI

from clarity.config import MODEL_FAST

logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format="%(name)s: %(message)s")
log = logging.getLogger("clarity.ask")
log.setLevel(logging.INFO)            # this module's messages, not every library's
client = OpenAI(timeout=60, max_retries=2)


class Truncated(RuntimeError):
    """The model ran out of room. The partial text is attached, not returned."""

    def __init__(self, partial: str):
        super().__init__("finish_reason was 'length'; the answer is incomplete")
        self.partial = partial


@dataclass(frozen=True)
class Answer:
    text: str
    model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


def ask(user: str, system: str | None = None, *, model: str = MODEL_FAST,
        max_tokens: int = 800, **params) -> Answer:
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": user})
    started = time.perf_counter()
    response = client.chat.completions.create(model=model, messages=messages,
                                              max_completion_tokens=max_tokens, **params)
    choice, usage = response.choices[0], response.usage
    answer = Answer(text=choice.message.content or "", model=response.model,
                    finish_reason=choice.finish_reason, prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    seconds=round(time.perf_counter() - started, 2))
    log.info("model=%s finish=%s in=%d out=%d %.2fs", answer.model, answer.finish_reason,
             answer.prompt_tokens, answer.completion_tokens, answer.seconds)
    if answer.finish_reason == "length":
        raise Truncated(answer.text)
    return answer


result = ask("Name Meridian Supply Co.'s likeliest three product categories, comma-separated.",
             system="You are terse.", temperature=0)
print("->", result.text)

try:
    ask("List twenty things a janitorial supplies distributor sells, one per line.",
        max_tokens=16)
except Truncated as error:
    print(f"-> Truncated: {error}; got {len(error.partial)} characters before the cut")
