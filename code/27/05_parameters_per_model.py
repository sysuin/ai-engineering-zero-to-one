# timeout: 300
# A gateway that hides the provider also has to hide the fact that models from one provider
# do not accept the same parameters. The same request, sent to both of Clarity's models.

from openai import BadRequestError, OpenAI

from clarity.config import CAPABILITIES, MODEL_FAST, MODEL_SMART

client = OpenAI()
REQUEST = {"messages": [{"role": "user", "content": "Reply with the single word: ready"}],
           "max_completion_tokens": 400, "temperature": 0, "seed": 7}


def send(model: str, request: dict) -> str:
    try:
        reply = client.chat.completions.create(model=model, **request)
        return f"ok: {(reply.choices[0].message.content or '').strip()[:20]!r}"
    except BadRequestError as error:
        body = error.body if isinstance(error.body, dict) else {}
        return f"rejected: {body.get('message', str(error)).split('. ')[0]}"


def for_model(model: str, request: dict) -> tuple[dict, list[str]]:
    """Keep what the model accepts; report what was dropped rather than hiding it."""
    allowed = CAPABILITIES.get(model, set()) | {"messages"}
    kept = {k: v for k, v in request.items() if k in allowed}
    return kept, sorted(set(request) - set(kept))


for model in (MODEL_FAST, MODEL_SMART):
    print(model)
    print(f"  sent as written          {send(model, REQUEST)}")
    kept, dropped = for_model(model, REQUEST)
    print(f"  filtered by capability   {send(model, kept)}"
          + (f"   (dropped: {', '.join(dropped)})" if dropped else ""))
