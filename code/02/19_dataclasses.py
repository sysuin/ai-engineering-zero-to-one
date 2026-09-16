# The one kind of class this book asks you to write: a dataclass, a record with named,
# typed fields.

import json
from dataclasses import FrozenInstanceError, asdict, dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    opened: date
    category: str
    body: str
    priority: str = "Normal"

    def age_days(self, today: date) -> int:
        """A method is a function that belongs to the record."""
        return (today - self.opened).days


lines = Path("data/meridian/documents/tickets/tickets.jsonl").read_text().splitlines()
raw = json.loads(lines[0])
ticket = Ticket(ticket_id=raw["ticket_id"], opened=date.fromisoformat(raw["opened"]),
                category=raw["category"], body=raw["body"], priority=raw["priority"])

print(ticket)
print("field access:", ticket.category, "|", ticket.priority)
print("age on 2025-12-31:", ticket.age_days(date(2025, 12, 31)), "days")
print("as a dictionary:", asdict(ticket)["ticket_id"], "...")
print("equal to a copy of itself:", ticket == Ticket(**asdict(ticket)))

# frozen=True: a record that cannot be changed after it is made.
try:
    ticket.priority = "Low"
except FrozenInstanceError as error:
    print("\nFrozenInstanceError:", error)

# What a dataclass does NOT do: check the types it declares. This is accepted silently.
nonsense = Ticket(ticket_id=42, opened="last Tuesday", category=None, body=["?"])
print("\naccepted without complaint:", nonsense)
print("-> Chapter 8's Pydantic models look the same and refuse this.")
