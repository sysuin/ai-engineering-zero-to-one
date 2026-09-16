# timeout: 1200
# A router, measured against one call that does everything. Two hundred tickets and eight
# messages that are not support tickets at all (written for this listing). Each design must
# name the ticket's kind and pull out the one field that kind needs.

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Union

from openai import OpenAI
from pydantic import BaseModel, Field

from clarity.config import MODEL_FAST

client = OpenAI()
rows = [json.loads(line) for line in
        Path("data/meridian/documents/tickets/tickets.jsonl").read_text().splitlines()][:200]
OFF_TOPIC = [
    "Do you have any job vacancies at the Columbus depot?",
    "Thanks, all sorted now!",
    "Our office will be closed on Friday for a staff event.",
    "Can I speak to someone about becoming a distributor?",
    "What is your company's environmental policy?",
    "Happy new year to the whole Meridian team!",
    "Please stop sending me the monthly newsletter.",
    "Is anyone going to the trade show in Chicago next month?",
]
KINDS = ("Delivery", "Quality", "Billing", "Returns", "Account")
FIELD = {"Delivery": "order_number", "Billing": "invoice_number",
         "Quality": "sku", "Returns": "sku", "Account": "request"}


def truth_field(kind: str, body: str) -> str | None:
    if kind == "Delivery":
        m = re.search(r"(?:order|commande)\s+#?(\d+)", body, re.I)
    elif kind == "Billing":
        m = re.search(r"(INV-\d+)", body)
    elif kind in ("Quality", "Returns"):
        m = re.search(r"(MRD-[A-Z]{3}-\d{3})", body)
    else:
        for request, cue in (("add_address", "address"), ("remove_user", "remove"),
                             ("pricing_copy", "pricing")):
            if cue in body.lower():
                return request
        return "other"
    return m.group(1) if m else None


# ---------------------------------------------------------------- the specialised prompts
class DeliveryRecord(BaseModel):
    order_number: str | None = Field(description="Digits only, if the ticket gives one.")


class BillingRecord(BaseModel):
    invoice_number: str | None = Field(description="Like INV-12345, if the ticket gives one.")


class ProductRecord(BaseModel):
    sku: str | None = Field(description="Like MRD-CLE-001, if the ticket gives one.")


class AccountRecord(BaseModel):
    request: Literal["add_address", "remove_user", "pricing_copy", "other"]


SPECIALIST = {
    "Delivery": ("A delivery complaint. Extract the order number.", DeliveryRecord),
    "Billing": ("A billing query. Extract the invoice number.", BillingRecord),
    "Quality": ("A product quality report. Extract the product code.", ProductRecord),
    "Returns": ("A return request. Extract the product code being returned.", ProductRecord),
    "Account": ("An account change request. Say which change is asked for.", AccountRecord),
}

CATEGORIES = ("Delivery: late, missing or partial orders. Quality: defective or changed products. "
              "Billing: invoices, tax, credit notes. Returns: sending goods back, collections, RMAs. "
              "Account: addresses, users, pricing documents.")


def router_model(escape: bool):
    kinds = KINDS + (("None of these",) if escape else ())

    class Route(BaseModel):
        kind: Literal[kinds]                                # type: ignore[valid-type]
    return Route


# ---------------------------------------------------------------- one call for everything
class Delivery(BaseModel):
    kind: Literal["Delivery"]
    order_number: str | None = Field(description="Digits only, if the ticket gives one.")


class Billing(BaseModel):
    kind: Literal["Billing"]
    invoice_number: str | None = Field(description="Like INV-12345, if the ticket gives one.")


class Quality(BaseModel):
    kind: Literal["Quality"]
    sku: str | None = Field(description="Like MRD-CLE-001, if the ticket gives one.")


class Returns(BaseModel):
    kind: Literal["Returns"]
    sku: str | None = Field(description="Like MRD-CLE-001, if the ticket gives one.")


class Account(BaseModel):
    kind: Literal["Account"]
    request: Literal["add_address", "remove_user", "pricing_copy", "other"]


class NotATicket(BaseModel):
    kind: Literal["None of these"]


class Ticket(BaseModel):
    issue: Union[Delivery, Billing, Quality, Returns, Account, NotATicket]


# The first draft of this listing built the branches by subclassing the specialist records,
# which put `kind` after the field in every branch except NotATicket. Kept, because of what it did.
class DeliveryLast(DeliveryRecord):
    kind: Literal["Delivery"]


class BillingLast(BillingRecord):
    kind: Literal["Billing"]


class QualityLast(ProductRecord):
    kind: Literal["Quality"]


class ReturnsLast(ProductRecord):
    kind: Literal["Returns"]


class AccountLast(AccountRecord):
    kind: Literal["Account"]


class TicketKindLast(BaseModel):
    issue: Union[DeliveryLast, BillingLast, QualityLast, ReturnsLast, AccountLast, NotATicket]


# calls, input tokens, output tokens
usage = {"router": [0, 0, 0], "single": [0, 0, 0], "kind last": [0, 0, 0], "forced": [0, 0, 0]}


def call(design: str, system: str, text: str, schema):
    reply = client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, response_format=schema, max_completion_tokens=120,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text}])
    u = usage[design]
    u[0] += 1
    u[1] += reply.usage.prompt_tokens
    u[2] += reply.usage.completion_tokens
    return reply.choices[0].message.parsed


def routed(text: str, escape: bool = True):
    route = call("router", f"Route this message. {CATEGORIES}",
                 text, router_model(escape))
    if route is None or route.kind not in SPECIALIST:
        return (route.kind if route else "None of these"), None
    instruction, schema = SPECIALIST[route.kind]    # a second call
    record = call("router", instruction, text, schema)
    field = getattr(record, FIELD[route.kind]) if record else None
    return route.kind, field


def single(text: str, schema=Ticket, design: str = "single") -> tuple[str, str | None]:
    parsed = call(design, "Classify the message and extract the field its kind calls for. "
                  + CATEGORIES + " Anything else is None of these.", text, schema)
    if parsed is None or parsed.issue.kind not in FIELD:
        return (parsed.issue.kind if parsed else "None of these"), None
    return parsed.issue.kind, getattr(parsed.issue, FIELD[parsed.issue.kind])


texts = [r["body"] for r in rows]
with ThreadPoolExecutor(max_workers=12) as pool:
    by_router = list(pool.map(routed, texts + OFF_TOPIC))
    by_single = list(pool.map(single, texts + OFF_TOPIC))
    kind_last = list(pool.map(lambda t: single(t, TicketKindLast, "kind last"), texts))
    no_escape = list(pool.map(lambda t: call("router", f"Route this message. {CATEGORIES}",
                                             t, router_model(False)), OFF_TOPIC))


def same(a: str | None, b: str | None) -> bool:
    return (a or "").upper() == (b or "").upper()


print(f"{len(rows)} tickets and {len(OFF_TOPIC)} messages that are not tickets\n")
print(f"  {'':<30}{'router + specialist':>20}{'one call':>10}")
for label, test in (
    ("kind matches the label", lambda out, r: out[0] == r["category"]),
    ("whole record right", lambda out, r: out[0] == r["category"]
     and same(out[1], truth_field(r["category"], r["body"]))),
):
    counts = [sum(test(o, r) for o, r in zip(outs[:len(rows)], rows))
              for outs in (by_router, by_single)]
    print(f"  {label:<30}{counts[0]:>20}{counts[1]:>10}")
counts = [sum(o[0] == "None of these" for o in outs[len(rows):]) for outs in (by_router, by_single)]
print(f"  {'non-tickets set aside':<30}{f'{counts[0]} of {len(OFF_TOPIC)}':>20}"
      f"{f'{counts[1]} of {len(OFF_TOPIC)}':>10}")
for i, label in ((0, "calls"), (1, "input tokens"), (2, "output tokens")):
    print(f"  {label:<30}{usage['router'][i]:>20,}{usage['single'][i]:>10,}")
kept = [t for t, o in zip(OFF_TOPIC, by_router[len(rows):]) if o[0] != "None of these"]
print("\n  non-tickets the router sent to a specialist:")
for t, o in zip(OFF_TOPIC, by_router[len(rows):]):
    if o[0] != "None of these":
        print(f"    {o[0]:<9} {t}")
forced = [route.kind if route else None for route in no_escape]
print(f"  with no 'None of these' class, the {len(OFF_TOPIC)} went to: "
      + ", ".join(f"{k} {forced.count(k)}" for k in sorted(set(filter(None, forced)))))

# A misroute, made on purpose: twenty tickets of one kind, sent to another kind's specialist.
examples: list[str] = []
print("\nTwenty tickets sent to the wrong specialist on purpose. What came back:\n")
print(f"  {'tickets':<9}{'sent to':<10}{'nothing':>9}{'a value from the ticket':>25}{'invented':>10}")
for kind, wrong in (("Account", "Delivery"), ("Delivery", "Billing"), ("Billing", "Quality")):
    group = [r for r in rows if r["category"] == kind][:20]
    instruction, schema = SPECIALIST[wrong]
    with ThreadPoolExecutor(max_workers=12) as pool:
        records = list(pool.map(lambda r: call("forced", instruction, r["body"], schema), group))
    values = [getattr(rec, FIELD[wrong]) if rec else None for rec in records]
    empty = sum(not v for v in values)
    copied = sum(bool(v) and v.upper() in r["body"].upper() for v, r in zip(values, group))
    print(f"  {kind:<9}{wrong:<10}{empty:>9}{copied:>25}{len(group) - empty - copied:>10}")
    example = next(((v, r) for v, r in zip(values, group) if v), None)
    if example:
        examples.append(f"    {wrong} specialist, {FIELD[wrong]} = {example[0]!r}: "
                        f"{example[1]['body'][:25]}…")
print("\n".join(examples))

right = sum(o[0] == r["category"] for o, r in zip(kind_last, rows))
print("\nThe one call again, `kind` last in every branch but 'None of these':")
print(f"  kind right on {right} of {len(rows)}; "
      f"'None of these' for {sum(o[0] == 'None of these' for o in kind_last)}")
