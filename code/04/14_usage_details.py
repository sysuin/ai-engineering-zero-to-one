# usage has more in it than three numbers. Two of the extra fields change what a call
# costs: tokens the provider did not have to recompute, and tokens you paid for and
# never saw.

from openai import OpenAI

from clarity.config import MODEL_FAST, MODEL_SMART

client = OpenAI()
QUESTION = ("A contract raises a unit price 12% in January and cuts it 12% in July. "
            "Is the July price higher, lower or the same as last December's? One line.")

for label, model in [("MODEL_FAST", MODEL_FAST), ("MODEL_SMART", MODEL_SMART)]:
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": QUESTION}])
    usage = response.usage
    details_in = usage.prompt_tokens_details
    details_out = usage.completion_tokens_details
    reasoning = details_out.reasoning_tokens or 0
    visible = usage.completion_tokens - reasoning
    print(label)
    print(f"  answer             {response.choices[0].message.content.strip()[:70]}")
    print(f"  prompt_tokens      {usage.prompt_tokens:>5}   of which cached "
          f"{details_in.cached_tokens or 0}")
    print(f"  completion_tokens  {usage.completion_tokens:>5}   of which reasoning "
          f"{reasoning}, visible {visible}")
    if usage.completion_tokens:
        print(f"  share of the output bill you never see: "
              f"{100 * reasoning / usage.completion_tokens:.0f}%")
    print()
