# A bug that happened one run in twenty did not happen in ten runs after the fix.
# Is it fixed? How many clean runs would it take to say so?

from math import ceil, log

print("  bug rate   runs needed to see it at least once, 95% of the time")
for rate in (0.50, 0.20, 0.05, 0.01):
    runs = ceil(log(0.05) / log(1 - rate))
    print(f"  {rate:>8.0%}   {runs:>6}")

print()
for clean in (10, 30, 100):
    chance = (1 - 0.05) ** clean
    print(f"  a 1-in-20 bug that was never fixed passes {clean:>3} runs in a row "
          f"{chance:.0%} of the time")

print("\nthe rule of three: after n clean runs, the bug rate is below about 3/n")
for n in (10, 60, 300):
    print(f"  {n:>3} clean runs -> rate under {3 / n:.1%}, 95% confidence")
