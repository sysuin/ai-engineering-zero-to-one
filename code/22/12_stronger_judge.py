# timeout: 2400
# Depends on clarity/evals/judge.py and code/22/_judge.json from 01_judge_basic.
# Which buys more: a stronger model as the judge, or the expected answer in the prompt?
# The same 215 answers with certain labels, graded by MODEL_SMART with and without the
# reference, set beside 01's MODEL_FAST verdicts.

import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.config import MODEL_SMART                                # noqa: E402
from clarity.evals.judge import POINTWISE, REFERENCE, Judge, kappa   # noqa: E402

rows = [r for r in json.load(open("code/22/_answerset.json"))
        if r["variant"] in ("plain", "wrong", "neighbour")]
fast = json.load(open("code/22/_judge.json"))
assert fast["ids"] == [r["id"] for r in rows], "01_judge_basic must be re-run first"
labels = [r["label"] for r in rows]
USAGE = {"prompt": 0, "completion": 0}


class SmartJudge(Judge):
    """The same prompts and parser; a reasoning model needs room to think first."""

    def _ask(self, user: str, tokens: int = 220) -> dict:
        reply = self.client.chat.completions.create(
            model=self.model, max_completion_tokens=4_000, reasoning_effort="low",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": self.system},
                      {"role": "user", "content": user}])
        USAGE["prompt"] += reply.usage.prompt_tokens
        USAGE["completion"] += reply.usage.completion_tokens
        try:
            return json.loads(reply.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return {"reason": "", "verdict": ""}


def grade(system: str) -> list[int]:
    judge = SmartJudge(model=MODEL_SMART, system=system)

    def one(row):
        expected = row["expected"] if system is REFERENCE else None
        verdict = judge.score(row["question"], row["answer"], expected)
        return 1 if verdict.correct else 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, rows))


def line(name: str, votes: list[int]) -> None:
    agreement, k = kappa(votes, labels)
    neighbours = [v for v, r in zip(votes, rows) if r["variant"] == "neighbour"]
    invented = [v for v, r in zip(votes, rows) if r["variant"] == "wrong"]
    good_failed = sum(1 for v, lab in zip(votes, labels) if lab and not v)
    print(f"  {name:28}{agreement:>6.0%}{k:>7.2f}{sum(invented):>6}/{len(invented)}"
          f"{sum(neighbours):>6}/{len(neighbours)}{good_failed:>6}/{sum(labels)}")


print(f"{len(rows)} answers, {sum(labels)} right and {len(rows) - sum(labels)} "
      "broken on purpose\n")
print(f"  {'judge':28}{'agree':>6}{'kappa':>7}{'invented':>9}{'neighbour':>10}{'right':>7}")
print(f"  {'':28}{'':>6}{'':>7}{'passed':>9}{'passed':>10}{'failed':>7}")
line("fast model, no reference", fast["no reference"]["votes"])
line("smart model, no reference", grade(POINTWISE))
line("fast model, with reference", fast["with the reference"]["votes"])
line("smart model, with reference", grade(REFERENCE))

calls = 2 * len(rows)
print(f"\nthe smart judge averaged {USAGE['completion'] / calls:,.0f} output tokens a verdict, "
      "its reasoning included,")
print(f"on prompts averaging {USAGE['prompt'] / calls:,.0f} tokens, the text the fast judge read")
