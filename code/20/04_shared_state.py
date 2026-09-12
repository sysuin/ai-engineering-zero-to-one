# timeout: 300
# The race you invent the moment two agents share a scratchpad.

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

WORKERS = 24
FINDINGS_EACH = 20


class Blackboard:
    """
    The shared-state pattern, written the way everybody writes it first.

    `self.facts + [fact]` builds a new list and rebinds the attribute. Between the read
    and the write, another thread can do the same thing from the same starting list,
    and one of the two findings is gone. No exception, no warning — just an agent's
    work that silently never happened.
    """

    def __init__(self) -> None:
        self.facts: list[str] = []

    def record(self, fact: str) -> None:
        current = self.facts
        time.sleep(0)                 # a scheduler yield; the API call is worse
        self.facts = current + [fact]


class LockedBlackboard(Blackboard):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def record(self, fact: str) -> None:
        with self._lock:
            super().record(fact)


def hammer(board: Blackboard) -> int:
    def work(worker: int) -> None:
        for n in range(FINDINGS_EACH):
            board.record(f"worker {worker} finding {n}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(work, range(WORKERS)))
    return len(board.facts)


expected = WORKERS * FINDINGS_EACH
naive = hammer(Blackboard())
locked = hammer(LockedBlackboard())

print(f"{WORKERS} workers, {FINDINGS_EACH} findings each — {expected} expected\n")
print(f"  shared list, no lock : {naive:>4} recorded, {expected - naive} lost")
print(f"  same code, with lock : {locked:>4} recorded, {expected - locked} lost")
print()
print("The API latency is taken out of that deliberately, so the bug happens every")
print("time instead of once a fortnight. With a real model call between the read and")
print("the write the window is a thousand times wider, and the failure is the same")
print("failure — you just meet it in production instead of here.")


# ---------------------------------------------------------------------------
# The second race, which a lock does not fix.
# ---------------------------------------------------------------------------
class Board:
    """A blackboard where agents check what has been done before doing it."""

    def __init__(self, lock: bool) -> None:
        self.done: dict[str, str] = {}
        self.work_done = 0
        self._lock = threading.Lock() if lock else None
        self._claims: set[str] = set()
        self.claiming = False

    def needs(self, key: str) -> bool:
        if self._lock is None:
            return key not in self.done
        with self._lock:
            if key in self.done or (self.claiming and key in self._claims):
                return False
            self._claims.add(key)
            return True

    def compute(self, key: str) -> None:
        if not self.needs(key):
            return
        time.sleep(0.05)              # the model call
        self.work_done += 1
        if self._lock:
            with self._lock:
                self.done[key] = "computed"
        else:
            self.done[key] = "computed"


def duplicate_test(lock: bool, claiming: bool) -> int:
    board = Board(lock)
    board.claiming = claiming
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(board.compute, ["revenue in 2024 Q3"] * 6))
    return board.work_done


print("\nSix agents, all asked to make sure revenue for 2024 Q3 is on the board.")
print("Each checks whether it is there before computing it.\n")
for label, lock, claiming in (("no lock", False, False),
                              ("a lock around every write", True, False),
                              ("a lock, and a claim before the work", True, True)):
    print(f"  {label:<36} {duplicate_test(lock, claiming)} of 6 actually computed it")
print()
print("A lock fixes corruption. It does not fix duplication, because all six agents")
print("read the board honestly, and all six were honestly told nothing was there —")
print("the work had started but not finished, and 'started' was not written down.")
print()
print("The fix is not a bigger lock. It is claiming the task before doing it, so the")
print("board records intent as well as results. That is the difference between a")
print("shared variable and a work queue, and it is the point at which most teams")
print("discover they have built a distributed system.")
