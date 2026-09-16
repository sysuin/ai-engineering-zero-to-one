# What a checksum buys a redaction filter, counted rather than asserted: how often a number
# that is not a card or an IBAN passes anyway, and which typing mistakes each check catches.

import random
import string

rng = random.Random(29)


def luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:                  # every second digit from the right
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def iban_ok(iban: str) -> bool:
    moved = iban[4:] + iban[:4]            # country and check digits go to the end
    return int("".join(str(int(ch, 36)) for ch in moved)) % 97 == 1


def card_prefix(digits: str) -> bool:      # Visa 4, Mastercard 51-55 and 2221-2720
    return digits[0] == "4" or 51 <= int(digits[:2]) <= 55 or 2221 <= int(digits[:4]) <= 2720


N = 200_000
numbers = ["".join(rng.choices(string.digits, k=16)) for _ in range(N)]
ibans = ["GB" + "".join(rng.choices(string.digits, k=2))
         + "".join(rng.choices(string.ascii_uppercase, k=4))
         + "".join(rng.choices(string.digits, k=14)) for _ in range(N)]

luhn_rate = sum(map(luhn, numbers)) / N
both_rate = sum(luhn(n) and card_prefix(n) for n in numbers) / N
iban_rate = sum(map(iban_ok, ibans)) / N
print(f"{N:,} random strings of the right shape; the share that passes anyway\n")
print(f"  sixteen digits, Luhn check                   {luhn_rate:>7.2%}")
print(f"  sixteen digits, Luhn check and card prefix   {both_rate:>7.2%}")
print(f"  GB IBAN shape, mod-97 check                  {iban_rate:>7.2%}")
print(f"\n  in a corpus with 10,000 sixteen-digit order numbers, false alarms expected:")
print(f"    pattern only {10_000:>7,}   + Luhn {luhn_rate * 10_000:>6,.0f}"
      f"   + Luhn and prefix {both_rate * 10_000:>5,.0f}")


def make_valid_card() -> str:
    body = "4" + "".join(rng.choices(string.digits, k=14))
    return body + next(d for d in string.digits if luhn(body + d))


def make_valid_iban() -> str:
    bban = "WEST" + "".join(rng.choices(string.digits, k=14))
    check = 98 - int("".join(str(int(ch, 36)) for ch in bban + "GB00")) % 97
    return f"GB{check:02d}{bban}"


def mistakes(value: str) -> tuple[list[str], list[str]]:
    """Every one-digit substitution, and every swap of two different neighbouring digits."""
    subs, swaps = [], []
    for i, ch in enumerate(value):
        if ch.isdigit():
            subs += [value[:i] + d + value[i + 1:] for d in string.digits if d != ch]
    for i in range(len(value) - 1):
        a, b = value[i], value[i + 1]
        if a.isdigit() and b.isdigit() and a != b:
            swaps.append(value[:i] + b + a + value[i + 2:])
    return subs, swaps


print("\nTyping mistakes in 2,000 valid values: the share each check rejects\n")
print(f"  {'':<22}{'one digit changed':>20}{'neighbours swapped':>20}")
for label, make, check in (("card number, Luhn", make_valid_card, luhn),
                           ("IBAN, mod 97", make_valid_iban, iban_ok)):
    rejected, total, missed_pairs = [0, 0], [0, 0], set()
    for _ in range(2_000):
        value = make()
        for kind, variants in enumerate(mistakes(value)):
            for v in variants:
                total[kind] += 1
                if check(v):
                    if kind == 1:
                        i = next(j for j in range(len(v)) if v[j] != value[j])
                        missed_pairs.add("".join(sorted(value[i:i + 2])))
                else:
                    rejected[kind] += 1
    print(f"  {label:<22}{rejected[0] / total[0]:>20.2%}{rejected[1] / total[1]:>20.2%}")
    if missed_pairs:
        print(f"    the swaps it accepted all exchanged: {', '.join(sorted(missed_pairs))}")
print(f"\n  exact counts behind the first table: Luhn {sum(map(luhn, numbers)):,}, "
      f"mod 97 {sum(map(iban_ok, ibans)):,} of {N:,}")
