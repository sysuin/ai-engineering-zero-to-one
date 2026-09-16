# timeout: 600
# What the provider's rate limiter thinks a request costs, read from the response headers, set
# beside what the same request is billed. A gateway that paces itself needs to know which it is.

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
PROMPTS = {
    "prose":        "Summarise this for a director. " + "Revenue grew in the Southwest while the Midwest "
                    "lost its largest account, and margin held close to target. " * 30,
    "a table":      "Which region grew most?\n" + "\n".join(f"| 2024 Q{q} | Region {r} | {1_000_000 + 7919 * q * r:,} |"
                                                              for q in range(1, 5) for r in range(1, 11)),
    "JSON":         "Validate this record. " + str([{"sku": f"MRD-CLE-{i:03}", "qty": i * 7, "price": 3.25 + i}
                                                       for i in range(60)]),
}


def one(text: str, ceiling: int) -> tuple[int, int, int]:
    raw = client.chat.completions.with_raw_response.create(
        model=MODEL_FAST, max_completion_tokens=ceiling, messages=[{"role": "user", "content": text}])
    reply = raw.parse()
    counted = int(raw.headers["x-ratelimit-limit-tokens"]) - int(raw.headers["x-ratelimit-remaining-tokens"])
    return reply.usage.prompt_tokens, reply.usage.completion_tokens, counted


print(f"  {'prompt':9}{'characters':>11}{'ceiling':>9}{'billed in':>10}{'billed out':>11}"
      f"{'limiter counted':>16}{'characters / 4':>15}")
for name, text in PROMPTS.items():
    for ceiling in (16, 4000):
        billed_in, billed_out, counted = one(text, ceiling)
        print(f"  {name:9}{len(text):>11,}{ceiling:>9,}{billed_in:>10,}{billed_out:>11,}{counted:>16,}{len(text) // 4:>15,}")

print("\n'limiter counted' is the drop in x-ratelimit-remaining-tokens against the limit, read from the")
print("response to that request; header names and behaviour are this provider's, today")
