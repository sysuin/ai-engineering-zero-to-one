# timeout: 900
# A question that needs arithmetic, answered two ways: directly, or by asking the model to
# write the calculation as a Python expression that code then evaluates.

import ast
import operator
import random
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
rng = random.Random(9)
N = 30

QUESTIONS = []
for _ in range(N):
    units, price = rng.randint(120, 980), round(rng.uniform(3, 90), 2)
    rise, discount = rng.choice([3, 5, 8, 12]), rng.choice([0, 5, 10, 15])
    answer = round(units * price * (1 + rise / 100) * (1 - discount / 100), 2)
    QUESTIONS.append((f"An order is for {units} units at ${price:.2f} each. The supplier applies "
                      f"a {rise}% price rise, then Meridian's {discount}% volume discount is taken "
                      "off the new total. What does the order cost, in dollars?", answer))

SAFE = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}


def evaluate(expression: str) -> float:
    """Arithmetic only: numbers and + - * /. Anything else is refused, not run."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE:
            return SAFE[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE:
            return SAFE[type(node.op)](walk(node.operand))
        raise ValueError(f"not allowed: {type(node).__name__}")
    return walk(ast.parse(expression, mode="eval"))


def direct(q: str) -> float | None:
    text = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=40,
        messages=[{"role": "user", "content": q + " Reply with the number only."}],
    ).choices[0].message.content or ""
    m = re.search(r"-?[\d,]+\.?\d*", text)
    return float(m.group().replace(",", "")) if m else None


def program(q: str) -> float | None:
    text = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=120,
        messages=[{"role": "user", "content": q + " Do not calculate it yourself. Reply with a "
                   "single Python arithmetic expression that computes it, and nothing else."}],
    ).choices[0].message.content or ""
    try:
        return evaluate(text.strip().strip("`").removeprefix("python").strip())
    except (ValueError, SyntaxError):
        return None


def right(value, answer) -> bool:
    return value is not None and abs(value - answer) < 0.01


results = {}
for name, fn in [("answer directly", direct), ("write an expression, code evaluates", program)]:
    with ThreadPoolExecutor(max_workers=10) as pool:
        values = list(pool.map(fn, [q for q, _ in QUESTIONS]))
    ok = sum(right(v, a) for v, (_, a) in zip(values, QUESTIONS))
    close = sum(v is not None and abs(v - a) / a < 0.01 for v, (_, a) in zip(values, QUESTIONS))
    results[name] = ok
    print(f"  {name:38} exact to the cent {ok:>2}/{N}   within 1% {close:>2}/{N}")

print(f"\nexample: {QUESTIONS[0][0][:88]}...")
print(f"         the right answer is ${QUESTIONS[0][1]:,.2f}")
