# timeout: 600
# Where the time and tokens of one answer go, stage by stage.

import sys
import time

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST        # noqa: E402
from clarity.prompts import load as load_prompt   # noqa: E402
from clarity.v0_5.rag import Clarity, _Grounding   # noqa: E402
from meridian_index import load_index         # noqa: E402

chunks, vectors = load_index()
clarity = Clarity(chunks, vectors)
QUESTION = "Which account did not renew, and what does the company say caused it?"

stages = []
t = time.perf_counter()
query = clarity._embed(QUESTION)
stages.append(("embed the question", time.perf_counter() - t, "—"))

t = time.perf_counter()
scores = clarity.vectors @ query
best = sorted(range(len(chunks)), key=lambda i: -scores[i])[:6]
stages.append((f"search {len(chunks):,} vectors", time.perf_counter() - t, "—"))
passages = clarity.retrieve(QUESTION, k=6)          # the same six, wrapped as Passages
context = clarity._context(passages)

t = time.perf_counter()
ground = clarity.client.chat.completions.parse(
    model=MODEL_FAST, temperature=0, max_completion_tokens=400,
    response_format=_Grounding,
    messages=[{"role": "system", "content": "Decide whether the excerpts answer the question. "
               "Being about the same subject is not enough."},
              {"role": "user", "content": f"<excerpts>\n{context}\n</excerpts>\n\nQuestion: {QUESTION}"}])
u = ground.usage
stages.append(("grounding check", time.perf_counter() - t, f"{u.prompt_tokens} in, {u.completion_tokens} out"))

t = time.perf_counter()
reply = clarity.client.chat.completions.create(
    model=MODEL_FAST, temperature=0, max_completion_tokens=500,
    messages=[{"role": "system", "content": load_prompt("answer")},
              {"role": "user", "content": f"<excerpts>\n{context}\n</excerpts>\n\nQuestion: {QUESTION}"}])
u = reply.usage
stages.append(("write the answer", time.perf_counter() - t, f"{u.prompt_tokens} in, {u.completion_tokens} out"))

total = sum(s for _, s, _ in stages)
print(f"  {'stage':24} {'seconds':>8} {'share':>6}   tokens")
for name, seconds, tokens in stages:
    print(f"  {name:24} {seconds:>8.3f} {seconds / total:>6.0%}   {tokens}")
print(f"  {'total':24} {total:>8.3f}")
slowest = max(stages, key=lambda s: s[1])[0]
print(f"\nthe slowest stage was: {slowest}")
