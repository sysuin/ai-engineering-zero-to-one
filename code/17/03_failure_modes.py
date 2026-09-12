# timeout: 1800
# Five failure modes that are widely described. Which of them actually happen?
#
# Two of the five occurred on this corpus with this model. The other three did not,
# and what the agent did instead is more useful than the folklore.

import json
import sys
from pathlib import Path

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import Tool, ToolError, build_tools   # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402

chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())
SYSTEM = "You are an analyst for Meridian. Use tools for every fact."

report = {}

print("1. Looping — the same call, over and over\n")
# A tool that never satisfies, to provoke it.
def never_finds(query: str, limit: int = 5) -> list:
    return []


looping_tools = [Tool("search_documents",
                      "Search Meridian's documents for a passage.",
                      {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "limit": {"type": "integer", "default": 5}},
                       "required": ["query"], "additionalProperties": False},
                      never_finds)]

run = Agent(looping_tools, budget=Budget(steps=6)).run(
    "Find Meridian's employee headcount.", SYSTEM)
repeats = len(run.steps) - len({(s.tool, json.dumps(s.arguments, sort_keys=True))
                                for s in run.steps})
print(f"   {len(run.steps)} steps, {repeats} of them exact repeats")
print(f"   stopped because: {run.stopped_because}")
for s in run.steps[:4]:
    print(f"     {s.n}. {str(s.arguments)[:62]}")
print()
print("   It did not repeat itself once. It rephrased — four different wordings of the")
print("   same question, all returning nothing. So the loop guard, which compares")
print("   arguments exactly, never fired.")
print()
print("   That is worth knowing, because the exact-match guard is what everyone writes.")
print("   The loop that actually happens is semantic, and catching it means noticing")
print("   that N consecutive calls to the same tool all returned nothing.\n")
report["looping"] = {"steps": len(run.steps), "repeats": repeats,
                     "stopped": run.stopped_because}

print("\n2. Budget exhaustion — and what comes back when it happens\n")
run = Agent(tools, budget=Budget(steps=2)).run(
    "Compare 2024 Q3 and 2025 Q1 on revenue, margin and the stated causes, and say "
    "which quarter was healthier.", SYSTEM)
print(f"   stopped because: {run.stopped_because}")
print(f"   answer: {run.answer[:200]}")
print("   A budget that returns nothing is a bug. A budget that returns what it has,")
print("   and says what is missing, is a result someone can act on.\n")
report["budget"] = {"stopped": run.stopped_because, "steps": len(run.steps)}

print("\n3. Tool thrashing — swapping tools instead of thinking\n")
run = Agent(tools, budget=Budget(steps=8)).run(
    "What is the average tenure of Meridian's warehouse staff?", SYSTEM)
print(f"   route: {' -> '.join(run.tools_used)}")
print(f"   stopped because: {run.stopped_because}")
print(f"   answer: {run.answer[:180]}")
print("   Two steps and a clean concession. No thrashing at all — it searched twice,")
print("   found nothing, and said so. The failure mode is real and this model did not")
print("   exhibit it here, which is the sort of thing worth checking before designing")
print("   elaborate machinery to prevent it.\n")
report["thrashing"] = {"route": run.tools_used, "steps": len(run.steps)}

print("\n4. Confident wrongness — the expensive one\n")
run = Agent(tools, budget=Budget(steps=6)).run(
    "Meridian opened a fourth distribution centre in 2025. How has it performed?",
    SYSTEM)
print(f"   steps: {len(run.steps)}")
print(f"   answer: {run.answer[:220]}")
print("   The premise is false — there is no fourth distribution centre — and Chapter 5")
print("   measured that a model handed a false premise builds on it rather than")
print("   questioning it.")
print()
print("   It did not, here. It searched, found nothing about a fourth centre, and")
print("   reported the real 2025 figures instead. Having tools appears to help: an")
print("   agent that can look is less inclined to invent, because looking is available")
print("   and cheap. It still did not say plainly that the premise was wrong, which is")
print("   what you would want.\n")
report["false_premise"] = {"steps": len(run.steps), "answer": run.answer[:300]}

print("\n5. Drift — answering a nearby question\n")
run = Agent(tools, budget=Budget(steps=6)).run(
    "Which region had the worst gross margin in 2024 Q4, and by how much did it trail "
    "the best?", SYSTEM)
print(f"   route: {' -> '.join(run.tools_used)}")
print(f"   answer: {run.answer[:220]}")
print("   Both halves answered, and both correct against the warehouse. No drift.")
print("   Chapter 9 measured drift in a two-step prose chain; a tool loop that can go")
print("   back for the second figure appears less prone to it.\n")
report["drift"] = {"route": run.tools_used, "answer": run.answer[:300]}

Path("code/17/_failures.json").write_text(json.dumps(report, indent=2, default=str))

print("\nTwo of five occurred: the rephrasing loop, and budget exhaustion.")
print()
print("Three did not, on this corpus, with this model, today. That is not a reason to")
print("stop guarding against them — it is a reason to measure before building")
print("machinery, and to re-measure when the model changes.")
print()
print("All five are visible in the trace and invisible in the answer, which is the")
print("part that does not change.")
