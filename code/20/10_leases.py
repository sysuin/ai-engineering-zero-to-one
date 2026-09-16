# A claim stops two agents doing the same work. It also stops anyone doing it if the
# agent holding the claim dies. A lease is a claim that expires.

import threading
import time

TASKS = ["2024 Q3", "2024 Q4", "2025 Q1", "2025 Q2"]
LEASE = 0.3                               # seconds a claim is honoured without renewal


class Board:
    def __init__(self, expires: bool):
        self.lock, self.expires = threading.Lock(), expires
        self.claims: dict[str, tuple[str, float]] = {}
        self.done: dict[str, str] = {}

    def claim(self, task: str, worker: str) -> bool:
        with self.lock:
            if task in self.done:
                return False
            holder = self.claims.get(task)
            if holder and not (self.expires and time.monotonic() - holder[1] > LEASE):
                return False              # someone holds it, and the claim still counts
            self.claims[task] = (worker, time.monotonic())
            return True

    def finish(self, task: str, worker: str) -> None:
        with self.lock:
            if self.claims.get(task, ("",))[0] == worker:
                self.done[task] = worker


def worker(board: Board, name: str, dies_on: str | None, rounds: int = 6) -> None:
    for _ in range(rounds):                   # keep coming back for unfinished work
        for task in TASKS:
            if board.claim(task, name):
                if task == dies_on:
                    return                    # crashed while holding the claim
                time.sleep(0.02)              # the work
                board.finish(task, name)
        time.sleep(0.1)


for expires in (False, True):
    board = Board(expires)
    crashing = threading.Thread(target=worker, args=(board, "agent-1", "2024 Q4"))
    crashing.start()
    crashing.join()                           # agent-1 claims 2024 Q3, then dies on Q4
    healthy = threading.Thread(target=worker, args=(board, "agent-2", None))
    healthy.start()
    healthy.join()
    missing = [t for t in TASKS if t not in board.done]
    label = "a lease that expires" if expires else "a claim that never expires"
    print(f"{label:28} done {len(board.done)} of {len(TASKS)}; "
          f"never done: {', '.join(missing) or 'none'}")
    for task, who in sorted(board.done.items()):
        print(f"    {task}  by {who}")
