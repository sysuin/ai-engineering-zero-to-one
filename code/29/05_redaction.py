# Personal data, found before the model sees it — and measured, because a detector that
# is never scored is a detector nobody knows the recall of.

import re

TEXT = [
    # (text, the spans a careful person would redact)
    ("Please call Dana on 07700 900123 or email dana.okafor@example.co.uk.",
     {"Dana", "07700 900123", "dana.okafor@example.co.uk"}),
    ("Card ending: 4539 1488 0343 6467 was declined on 14 March.",
     {"4539 1488 0343 6467"}),
    ("Order 4539 1488 0343 6468 shipped; invoice INV-2024-00481.", set()),
    ("Refund to GB82 WEST 1234 5698 7654 32, reference MSC-2024-118.",
     {"GB82 WEST 1234 5698 7654 32"}),
    ("Pallet count 1234 5678 9012 3456 confirmed at the Leeds depot.", set()),
    ("Supplier contact: ops@vosss-industrial.example, tel +44 20 7946 0958.",
     {"ops@vosss-industrial.example", "+44 20 7946 0958"}),
    ("Margin fell 0.3 points; see the Q1 review, page 12.", set()),
    ("The new account is GB29 NWBK 6016 1331 9268 19 for Halloway Group.",
     {"GB29 NWBK 6016 1331 9268 19"}),
]

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?:\+44\s?\d{2}|\b0\d{4})[\s\d]{6,11}\d")
CARD = re.compile(r"\b(?:\d[ -]?){15}\d\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,4})?\b")


def luhn(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2)
                for i, d in enumerate(digits))
    return total % 10 == 0


def iban_ok(value: str) -> bool:
    compact = value.replace(" ", "")
    moved = compact[4:] + compact[:4]
    return int("".join(str(int(ch, 36)) for ch in moved)) % 97 == 1


def detect(text: str, checksums: bool) -> set[str]:
    found = set(EMAIL.findall(text)) | {m.strip() for m in PHONE.findall(text)}
    found |= {m for m in CARD.findall(text) if not checksums or luhn(m)}
    found |= {m for m in IBAN.findall(text) if not checksums or iban_ok(m)}
    return found


print(f"{len(TEXT)} sentences, {sum(len(g) for _, g in TEXT)} spans a person would redact\n")
print(f"  {'detector':<28} {'found':>6} {'missed':>7} {'false alarms':>13}")
for label, checksums in (("patterns only", False), ("patterns + checksums", True)):
    hits = misses = false = 0
    for text, gold in TEXT:
        found = detect(text, checksums)
        hits += len(found & gold)
        misses += len(gold - found)
        false += len(found - gold)
    print(f"  {label:<28} {hits:>6} {misses:>7} {false:>13}")

# Pseudonymise rather than delete: the same value gets the same placeholder, so the model
# can still say "the card was declined twice", and the mapping stays on our side.
mapping: dict[str, str] = {}


def pseudonymise(text: str) -> str:
    for value in sorted(detect(text, checksums=True), key=len, reverse=True):
        kind = ("EMAIL" if "@" in value else "IBAN" if value[:2].isalpha()
                else "CARD" if CARD.fullmatch(value) else "PHONE")
        if value not in mapping:
            same_kind = sum(1 for p in mapping.values() if p.startswith(f"<{kind}_"))
            mapping[value] = f"<{kind}_{same_kind + 1}>"
        text = text.replace(value, mapping[value])
    return text


print("\n  what the model is sent:")
for text, _ in TEXT[:2] + TEXT[3:4]:
    print(f"    {pseudonymise(text)}")
print(f"\n  {len(mapping)} values kept in a mapping the model never sees; the name in")
print("  the first sentence is not among them, because no pattern knows what a name")
print("  looks like")
