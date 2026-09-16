# timeout: 180
# The arithmetic tool refuses everything but numbers and + - * / **, and its tests pass. A
# handful of hostile expressions — the kind a confused model or an injected document produces —
# through Clarity's tool runner, against the tool as this chapter first wrote it and as it is now.

import ast
import dataclasses
import inspect
import operator
import random
import subprocess
import sys
import textwrap
import threading
import time

sys.path.insert(0, "code")
from clarity.v0_8.tools import Tool, ToolError, ToolRunner, build_tools  # noqa: E402

ALLOWED = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def first_version(expression: str) -> float:
    """The walk as Chapter 16 first wrote it: the grammar is right, the sizes are not checked."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED:
            return ALLOWED[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED:
            return ALLOWED[type(node.op)](walk(node.operand))
        raise ToolError(f"{type(node).__name__} is not allowed")
    return round(float(walk(ast.parse(expression, mode="eval").body)), 6)


class Nothing:                                   # no documents and no warehouse are needed
    last_refusal = ""

    def search(self, query, k=5):
        return []

    def ask(self, question):
        return None


now = next(t for t in build_tools(Nothing(), Nothing()) if t.name == "arithmetic")
VERSIONS = {"first version": Tool("arithmetic", "", {}, first_version, timeout=1.0),
            "now": dataclasses.replace(now, timeout=1.0)}


def execute(tool: Tool, expression: str) -> tuple[str, float]:
    """The runner's own step for one call, without a model: what the model would read back."""
    runner = ToolRunner([tool], client=object())
    result, seconds, _ = runner._execute("arithmetic", {"expression": expression})
    return result, seconds


def show(version: str, text: str) -> None:
    lines = textwrap.wrap(text, 58)
    print(f"    {version:<14} " + f"\n    {'':<14} ".join(lines))


CASES = ["(8461842 - 5661205) / 8461842", "1.01 ** 365", "1 / 0", "(-8) ** 0.5",
         "2.0 ** 2000", "1e200 * 1e200", "1" + " + 1" * 2_000]
print("What the model reads back, with each tool's timeout set to one second:")
for expression in CASES:
    label = expression if len(expression) <= 40 else f"{expression[:20]}… ({len(expression):,} characters)"
    print(f"\n  {label}")
    for version, tool in VERSIONS.items():
        show(version, execute(tool, expression)[0])


def longest_stall(tool: Tool, text: str) -> tuple[str, float, float]:
    """One call, while another thread ticks every 50ms: the longest gap."""
    ticks, done = [time.perf_counter()], threading.Event()

    def tick():
        while not done.is_set():
            time.sleep(0.05)
            ticks.append(time.perf_counter())

    ticker = threading.Thread(target=tick)
    ticker.start()
    result, seconds = execute(tool, text)
    done.set()
    ticker.join()
    return result, seconds, max(b - a for a, b in zip(ticks, ticks[1:]))


print("\n7 ** 10 ** 7 through the runner, while another thread ticks every 50ms:")
for version, tool in VERSIONS.items():
    result, seconds, stall = longest_stall(tool, "7 ** 10 ** 7")
    print(f"\n    {version:<14} returned after {seconds:.2f}s; the ticks stopped for {stall:.2f}s")
    show("", result)

print("\nHow long the first version holds the interpreter to raise 7 to a power:")
timings = []
for power in (5, 6, 7):
    started = time.perf_counter()
    try:
        first_version(f"7 ** 10 ** {power}")
    except OverflowError:
        pass
    timings.append(time.perf_counter() - started)
    print(f"  7 ** 10 ** {power:<3} {timings[-1]:>7.2f}s")
growth = timings[2] / timings[1]
print(f"  each tenfold exponent took about {growth:.0f} times longer; at that rate")
print(f"  7 ** 10 ** 8 would hold it for roughly {timings[2] * growth / 60:.0f} minutes, and no timeout")
print("  in the same process could fire until it let go")


# A fuzz test proper: expressions nobody chose, thrown at the tool as it is now.
rng = random.Random(16)
LEAVES = ["0", "-1", "1e300", "0.5"]


def generate(depth: int = 0) -> str:
    if depth > 4 or rng.random() < 0.3:
        if rng.random() < 0.5:
            return rng.choice(LEAVES)
        return str(rng.randint(0, 10 ** rng.randint(1, 30)))
    if rng.random() < 0.1:
        return f"-{generate(depth + 1)}"
    op = rng.choice(["+", "-", "*", "/", "**"])
    left, right = generate(depth + 1), generate(depth + 1)
    return f"({left} {op} {right})"


EXPRESSIONS = [generate() for _ in range(20_000)]
outcomes, slowest = {"computed": 0, "refused": 0}, 0.0
for text in EXPRESSIONS:
    started = time.perf_counter()
    try:
        now.run(expression=text)
        outcomes["computed"] += 1
    except ToolError:
        outcomes["refused"] += 1
    except Exception as error:                   # noqa: BLE001
        name = type(error).__name__
        outcomes[name] = outcomes.get(name, 0) + 1
    slowest = max(slowest, time.perf_counter() - started)
escaped = sum(v for k, v in outcomes.items() if k not in ("computed", "refused"))
print("\n20,000 random expressions against the tool as it is now:")
print(f"  computed {outcomes['computed']:,}, refused {outcomes['refused']:,}, "
      f"raised anything else {escaped}")
print(f"  slowest single call {slowest * 1000:.0f}ms")

# The same expressions against the first version, in a process of its own: a thread cannot
# be stopped, but a process can be killed.
CHILD = f"""
import ast, operator, sys
class ToolError(Exception): pass
ALLOWED = {{ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}}
{inspect.getsource(first_version)}
for number, line in enumerate(sys.stdin, 1):
    print(number, flush=True)
    try:
        first_version(line)
    except Exception:
        pass
print("finished", flush=True)
"""
DEADLINE = 10
try:
    child = subprocess.run([sys.executable, "-c", CHILD], input="\n".join(EXPRESSIONS),
                           capture_output=True, text=True, timeout=DEADLINE)
    print("\nThe first version finished all 20,000.")
except subprocess.TimeoutExpired as stuck:
    seen = stuck.stdout or b""
    seen = seen.decode() if isinstance(seen, bytes) else seen
    number = int(seen.split()[-1])
    print(f"\nThe first version, given the same expressions and killed after {DEADLINE}s,")
    print(f"was still computing number {number}:")
    for line in textwrap.wrap(EXPRESSIONS[number - 1], 70):
        print("  " + line)
