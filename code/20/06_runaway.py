# timeout: 3000
# A failure mode one agent cannot have: agents that can call each other.

import json
import sys

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import Tool, _obj, build_tools   # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402

QUESTION = ("Why did gross margin fall in 2025 Q1, and by how much, and is that "
            "consistent with what the supplier contracts allow?")
DEPTH_LIMIT = 2
TOKEN_CEILING = 40_000

chunks, vectors = load_index()
base = {t.name: t for t in build_tools(Retriever(chunks, vectors), Warehouse())}


def mutual_team(cap: int | None, miswired: bool = False) -> dict:
    """
    Two colleagues who can each ask the other. Nothing here is unreasonable — it is
    exactly what "let the specialists collaborate" means when you write it down.

    The unbounded arm still has a token ceiling, because a book that re-runs its own
    listings should not contain one that decides how much it feels like spending.
    """
    stats = {"calls": 0, "tokens": 0, "depth": 0, "hops": [], "hit_ceiling": False}

    ROLES = {
        "researcher": (["search_documents"],
                       "You are Meridian's researcher. You read documents. For any "
                       "figure, ask the analyst."),
        "analyst": (["query_warehouse", "arithmetic"],
                    "You are Meridian's analyst. You compute figures. For any "
                    "context or cause, ask the researcher."),
    }
    OTHER = {"researcher": "analyst", "analyst": "researcher"}

    def tools_for(who: str, depth: int) -> list[Tool]:
        names, _ = ROLES[who]
        own = [base[n] for n in names]
        return own + [delegation(who, OTHER[who], depth)]

    def delegation(caller: str, target: str, depth: int) -> Tool:
        def ask(question: str) -> str:
            if cap is not None and depth >= cap:
                return ("Delegation limit reached. Answer from what you have and say "
                        "what is still unchecked.")
            if stats["tokens"] >= TOKEN_CEILING:
                stats["hit_ceiling"] = True
                return "Token ceiling reached. Answer with what you have."
            stats["depth"] = max(stats["depth"], depth + 1)
            stats["hops"].append(f"{'  ' * min(depth, 6)}{caller} -> {target}")
            worker = Agent(tools_for(target, depth + 1), budget=Budget(steps=3))
            # The slip: brief the colleague with the *caller's* role. One wrong
            # subscript, and each worker is now talking to a copy of itself.
            briefing = ROLES[caller if miswired and caller in ROLES else target][1]
            run = worker.run(question, system=briefing)
            stats["calls"] += len(run.steps) + 1
            stats["tokens"] += run.tokens
            return run.answer

        return Tool(f"ask_{target}", f"Ask the {target} a question in their area.",
                    _obj(question={"type": "string", "description": "A question."}),
                    ask, timeout=300)

    boss = Agent([delegation("supervisor", "researcher", 0),
                  delegation("supervisor", "analyst", 0)], budget=Budget(steps=6))
    run = boss.run(QUESTION, system="You lead the team. Delegate, then answer.")
    stats["calls"] += len(run.steps) + 1
    stats["tokens"] += run.tokens
    stats["answer"] = run.answer
    return stats


TRIALS = 2

print("Two specialists who can each consult the other. No cycle is written down;")
print("the cycle is what 'let them collaborate' means once it is code.")
print(f"\nEach unbounded arrangement is run {TRIALS} times, because one run of this")
print("measures the day rather than the design.\n")

results = {}
for label, cap, miswired in (("as intended", None, False),
                             ("one wrong subscript", None, True)):
    runs = [mutual_team(cap, miswired) for _ in range(TRIALS)]
    depths = sorted(r["depth"] for r in runs)
    tokens = sorted(r["tokens"] for r in runs)
    results[label] = {"depths": depths, "tokens": tokens,
                      "ceiling": sum(r["hit_ceiling"] for r in runs),
                      "hops": runs[depths.index(max(depths))]["hops"][:6]}
    print(f"{label}, no limit of any kind")
    print(f"  depth  : {depths}")
    print(f"  tokens : {[f'{t:,}' for t in tokens]}")
    if results[label]["ceiling"]:
        print(f"  {results[label]['ceiling']} of {TRIALS} runs were stopped by the "
              f"{TOKEN_CEILING:,}-token ceiling,")
        print(f"    which is this listing's safety net and not part of the design")
    print()

capped = mutual_team(DEPTH_LIMIT, False)
results["capped"] = {"depths": [capped["depth"]], "tokens": [capped["tokens"]],
                     "ceiling": 0, "hops": capped["hops"][:6]}
print(f"as intended, with a depth limit of {DEPTH_LIMIT}")
print(f"  depth  : {capped['depth']}")
print(f"  tokens : {capped['tokens']:,}")
print(f"  chain  :")
for hop in capped["hops"][:5]:
    print(f"      {hop}")
print()

json.dump(results, open("code/20/_runaway.json", "w"), indent=1)

worst = max(max(r["depths"]) for k, r in results.items() if k != "capped")
worst_tokens = max(max(r["tokens"]) for k, r in results.items() if k != "capped")
best_tokens = min(min(r["tokens"]) for k, r in results.items() if k != "capped")

print(f"Across {TRIALS * 2} runs of two arrangements that draw as the same picture, the")
print(f"deepest chain was {worst} agents and the most expensive answer cost "
      f"{worst_tokens:,} tokens —")
print(f"against {best_tokens:,} for the cheapest. Same question, every time.")
print()
print("The miswiring is one subscript: brief the colleague with the caller's role")
print("instead of its own, and each worker ends up consulting a copy of itself.")
print()
worst_label = max((k for k in results if k != "capped"),
                  key=lambda k: max(results[k]["depths"]))
inner = max(results[k]["depths"][-1] - results[k]["depths"][0]
            for k in results if k != "capped")
print(f"Today that arrangement — {worst_label} — went deepest. Do not read much into")
print(f"which one it was: the same arrangement varied by {inner} levels of depth "
      f"across its")
print(f"own {TRIALS} runs, on the same question, with nothing changed between them.")
print()
print("That is the lesson, and it is not really about recursion. Neither unbounded")
print("arrangement decided to stop; each one happened to. A run that finishes cheaply")
print("is not evidence about the next one, so there is no version of 'we tested it")
print("and it was fine' that covers this.")
print()
print("Only the last arrangement is bounded, and bounded is the only kind you can run")
print("without watching it.")
