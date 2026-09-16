# timeout: 900
# Four tools, one tool with an action field, or one tool with a nested object per action.
# The same twelve requests through each, graded on the arguments that matter.

import json
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
RUNS = 2
STATUS = ["open", "shipped", "delivered", "cancelled"]

# (request, action, the arguments that must be right)
REQUESTS = [
    ("Look up the customer called Halloway Group.", "find_customer", {"name": "Halloway Group"}),
    ("Find Quillon Metals in the CRM.", "find_customer", {"name": "Quillon Metals"}),
    ("Please find the customer Pemberton Mills.", "find_customer", {"name": "Pemberton Mills"}),
    ("Show open orders for customer C-1043.", "list_orders", {"customer_id": "C-1043", "status": "open"}),
    ("List every order for customer C-2210.", "list_orders", {"customer_id": "C-2210", "status": None}),
    ("Which of C-0087's orders have shipped?", "list_orders", {"customer_id": "C-0087", "status": "shipped"}),
    ("Cancelled orders for C-1043?", "list_orders", {"customer_id": "C-1043", "status": "cancelled"}),
    ("Refund 40.00 on order O-55120, the goods arrived damaged.", "issue_refund",
     {"order_id": "O-55120", "amount": 40.0}),
    ("Customer was double charged 129.50 on order O-77310 — refund it.", "issue_refund",
     {"order_id": "O-77310", "amount": 129.5}),
    ("Give back 15 on O-10044; late delivery.", "issue_refund", {"order_id": "O-10044", "amount": 15.0}),
    ("C-3301 has moved to 14 Harbour Road, Leeds LS1 4AB.", "update_address",
     {"customer_id": "C-3301", "address": "LS1 4AB"}),
    ("Change the address for C-0452 to Unit 7, Riverside Park, Dallas TX 75201.", "update_address",
     {"customer_id": "C-0452", "address": "75201"}),
]

FIELDS = {"find_customer": {"name": {"type": "string"}},
          "list_orders": {"customer_id": {"type": "string"},
                          "status": {"type": ["string", "null"], "enum": STATUS + [None],
                                     "description": "null for every order"}},
          "issue_refund": {"order_id": {"type": "string"}, "amount": {"type": "number"},
                           "reason": {"type": "string"}},
          "update_address": {"customer_id": {"type": "string"}, "address": {"type": "string"}}}
DOING = {"find_customer": "Find a customer by name.", "list_orders": "List a customer's orders.",
         "issue_refund": "Refund an amount on one order.", "update_address": "Change a customer's address."}


def obj(properties):
    return {"type": "object", "additionalProperties": False, "required": list(properties),
            "properties": properties}


def function(name, description, parameters):
    return {"type": "function", "function": {"name": name, "strict": True, "description": description,
                                             "parameters": parameters}}


SEPARATE = [function(n, DOING[n], obj(f)) for n, f in FIELDS.items()]
every = {k: v for f in FIELDS.values() for k, v in f.items()}
FLAT = [function("crm", "Customer records: " + " ".join(f"{n}: {d}" for n, d in DOING.items()), obj(
    {"action": {"type": "string", "enum": list(FIELDS)},
     **{k: {**v, "type": [v["type"], "null"] if isinstance(v["type"], str) else v["type"],
            **({"enum": v["enum"]} if "enum" in v else {})} for k, v in every.items()}}))]
NESTED = [function("crm", "Customer records. Put the request for exactly one action in 'request'.", obj(
    {"request": {"anyOf": [obj({"action": {"type": "string", "enum": [n]}, **f})
                           for n, f in FIELDS.items()]}}))]
DESIGNS = {"four tools": SEPARATE, "one tool, action field": FLAT, "one tool, nested request": NESTED}


def as_action(design: str, call) -> tuple[str | None, dict]:
    args = json.loads(call.function.arguments)
    if design == "four tools":
        return call.function.name, args
    if design == "one tool, nested request":
        args = args.get("request", {})
    return args.get("action"), args


def matches(args: dict, want: dict) -> bool:
    for key, value in want.items():
        got = args.get(key)
        if key in ("name", "address"):
            ok = isinstance(got, str) and value.lower() in got.lower()
        elif key == "amount":
            ok = isinstance(got, (int, float)) and abs(got - value) < 1e-9
        else:
            ok = got == value
        if not ok:
            return False
    return True


def fake_result(action: str | None, args: dict) -> dict:
    """What a real CRM would say: a lookup by id returns that id, by name a new one."""
    if action == "find_customer":
        name = str(args.get("name", ""))
        return {"customer_id": name if name.startswith("C-") else "C-9000", "status": "active"}
    return {"list_orders": {"orders": []}, "issue_refund": {"refunded": True},
            "update_address": {"updated": True}}.get(action, {"error": "unknown action"})


def grade(design: str, request: str, action: str, want: dict) -> tuple[bool, int, str]:
    """Run up to three rounds with fake results, so a model that looks the customer up first
    and then acts is not marked wrong for planning."""
    messages = [{"role": "system", "content": "Carry out the request with the tools."},
                {"role": "user", "content": request}]
    calls, done = [], False
    for _ in range(3):
        reply = client.chat.completions.create(
            model=MODEL_FAST, temperature=0, max_completion_tokens=300, tools=DESIGNS[design],
            messages=messages).choices[0].message
        if not reply.tool_calls:
            break
        messages.append(reply)
        for call in reply.tool_calls:
            got_action, args = as_action(design, call)
            calls.append(got_action)
            done |= got_action == action and matches(args, want)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(fake_result(got_action, args))})
        if done:
            break
    return done, len(calls), " -> ".join(str(c) for c in calls)


jobs = [(d, *r) for d in DESIGNS for r in REQUESTS for _ in range(RUNS)]
with ThreadPoolExecutor(max_workers=12) as pool:
    graded = list(pool.map(lambda job: (job, grade(*job)), jobs))

print(f"{len(REQUESTS)} requests x {RUNS} runs, up to three rounds with fake results\n")
print(f"  {'design':26}{'done right':>11}{'calls':>7}{'schema tokens':>15}")
for design, tools in DESIGNS.items():
    mine = [(job, g) for job, g in graded if job[0] == design]
    tokens = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=16, tools=tools,
        messages=[{"role": "user", "content": "hi"}]).usage.prompt_tokens
    print(f"  {design:26}{sum(g[0] for _, g in mine):>8}/{len(mine)}{sum(g[1] for _, g in mine):>7}{tokens:>15}")
    for (_, request, *_), (ok, n, path) in mine:
        if not ok or n > 1:
            print(f"      {'done' if ok else 'NOT DONE':8} {request[:44]:45} {path}")
