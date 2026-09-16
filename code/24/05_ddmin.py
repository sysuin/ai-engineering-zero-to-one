# Delta debugging done properly. Dropping one part at a time costs a test per part and
# finds a failure that needs one part. Zeller's ddmin splits instead, and finds a failure
# that needs two parts together — measured on a predicate with a known answer.

CAUSE = {11, 42}                      # the bug needs both of these passages together
tests = {"n": 0}


def fails(parts: list[int]) -> bool:
    tests["n"] += 1
    return CAUSE <= set(parts)


def drop_one_at_a_time(parts: list[int]) -> list[int]:
    kept = list(parts)
    for part in parts:
        trial = [p for p in kept if p != part]
        if fails(trial):
            kept = trial
    return kept


def ddmin(parts: list[int]) -> list[int]:
    """Zeller and Hildebrandt's minimising delta debugging, in its plain form."""
    n = 2
    while len(parts) >= 2:
        chunk = len(parts) // n
        subsets = [parts[i:i + chunk] for i in range(0, len(parts), chunk)]
        reduced = False
        for subset in subsets:                         # does one piece alone fail?
            if fails(subset):
                parts, n, reduced = subset, 2, True
                break
        if not reduced:
            for subset in subsets:                     # does leaving one piece out still fail?
                rest = [p for p in parts if p not in subset]
                if fails(rest):
                    parts, n, reduced = rest, max(n - 1, 2), True
                    break
        if not reduced:
            if n >= len(parts):
                break
            n = min(len(parts), n * 2)
    return parts


for size in (64, 256, 1024):
    parts = list(range(size))
    for name, method in (("one at a time", drop_one_at_a_time), ("ddmin", ddmin)):
        tests["n"] = 0
        found = method(parts)
        print(f"  {size:>5} parts  {name:14} {tests["n"]:>5} tests   left {found}")
    print()

RUNS = 3                              # each test of a flaky system is several model calls
print(f"at {RUNS} runs per test, 256 parts cost {RUNS * (256):,} model calls one at a time;")
tests["n"] = 0
ddmin(list(range(256)))
print(f"ddmin needed {RUNS * tests['n']:,}")
