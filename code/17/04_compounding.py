# Why long runs fail: per-step reliability compounds. And why agents are not quite as
# bad as the arithmetic says: a step that goes wrong can sometimes be noticed and fixed.

STEPS = [1, 5, 10, 20, 40]


def success(p: float, steps: int, recovery: float = 0.0) -> float:
    """Chance a run of `steps` ends right if each step is right with probability p,
    and a wrong step is noticed and repaired with probability `recovery`."""
    per_step = p + (1 - p) * recovery
    return per_step ** steps


print("chance the whole run is right, if every step must be\n")
print(f"  {'per step':>9}" + "".join(f"{n:>8} {'step' if n == 1 else 'steps'}"
                                    for n in STEPS))
for p in (0.99, 0.95, 0.90):
    print(f"  {p:>9.0%}" + "".join(f"{success(p, n):>13.1%}" for n in STEPS))

print("\nthe same, if half the wrong steps are noticed and repaired\n")
for p in (0.99, 0.95, 0.90):
    print(f"  {p:>9.0%}" + "".join(f"{success(p, n, 0.5):>13.1%}" for n in STEPS))

needed = next(n for n in range(1, 1000) if success(0.95, n) < 0.5)
print(f"\nAt 95% a step, a run of {needed} steps is more likely wrong than right.")
print("Recovery helps only if errors are visible to the agent — which is the argument")
print("for tools whose results can be checked, and errors that say what went wrong.")
