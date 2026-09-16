# timeout: 300
# A deploy stops the old process while it is answering. Five slow requests in flight, a SIGTERM
# one second in, and a server given one, three or eight seconds to finish what it started.

import signal
import socket
import subprocess
import sys
import threading
import time

import httpx

WORK = 4.0                                     # seconds: a model-backed answer


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def trial(grace: float) -> tuple[str, str]:
    port = free_port()
    server = subprocess.Popen([sys.executable, "code/25/_graceful_child.py", str(port),
                               str(grace), str(WORK)], stderr=subprocess.DEVNULL)
    for _ in range(200):                        # wait until it accepts connections
        try:
            httpx.get(f"http://127.0.0.1:{port}/ready", timeout=0.2)
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    outcomes: list[str] = []

    def call() -> None:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/ask", timeout=20)
            outcomes.append("answered" if r.status_code == 200 else f"a {r.status_code}")
        except httpx.HTTPError as error:
            outcomes.append(f"connection {type(error).__name__}")

    threads = [threading.Thread(target=call) for _ in range(5)]
    for t in threads:
        t.start()
    time.sleep(1.0)
    server.send_signal(signal.SIGTERM)           # what an orchestrator sends first
    time.sleep(0.2)
    try:
        late = f"a {httpx.get(f'http://127.0.0.1:{port}/ready', timeout=2).status_code}"
    except httpx.HTTPError as error:
        late = f"refused ({type(error).__name__})"
    for t in threads:
        t.join()
    server.wait(timeout=30)
    counts = {o: outcomes.count(o) for o in sorted(set(outcomes))}
    return ", ".join(f"{n} {o}" for o, n in counts.items()), late


print(f"five requests of {WORK:.0f}s each in flight; SIGTERM sent 1s after they started\n")
print(f"  {'grace':>6}  {'the five in flight':38}{'a new request, 0.2s after SIGTERM'}")
for grace in (1, 2, 5):
    in_flight, late = trial(grace)
    print(f"  {grace:>5}s  {in_flight:38}{late}")
