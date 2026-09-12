"""
Clarity v0.18 — the four patterns that keep one bad dependency from becoming an outage.

Each is a few dozen lines and each exists because of a specific failure that a retry
loop makes worse rather than better.

    retry with jitter   a provider blips; a naive retry turns one blip into a stampede
    timeout             a request that never returns holds a worker forever
    circuit breaker     a provider is down; retrying is just a slower way to fail
    bulkhead            one slow dependency should not consume every worker you have

The order matters. A timeout without a retry is an error budget you spend on nothing;
a retry without a circuit breaker is an outage amplifier; a circuit breaker without a
bulkhead protects the caller and not the process.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


class Transient(Exception):
    """Worth retrying. Anything else is not, and that judgement is the hard part."""


@dataclass
class Retry:
    attempts: int = 4
    base: float = 0.2
    cap: float = 4.0
    jitter: bool = True

    def __call__(self, fn: Callable, *args, **kwargs):
        for attempt in range(self.attempts):
            try:
                return fn(*args, **kwargs)
            except Transient:
                if attempt == self.attempts - 1:
                    raise
                delay = min(self.cap, self.base * (2 ** attempt))
                # Without jitter, every client that failed at the same moment retries
                # at the same moment, and the recovering provider is hit by the whole
                # fleet in lockstep. Full jitter is the version that works.
                time.sleep(random.uniform(0, delay) if self.jitter else delay)
        raise RuntimeError("unreachable")


class Open(Exception):
    """The breaker is open: fail immediately rather than slowly."""


@dataclass
class CircuitBreaker:
    """
    Three states, and the half-open one is the whole design.

        closed      calls pass through; consecutive failures are counted
        open        calls fail immediately, for `cooldown` seconds
        half-open   one call is allowed through to find out

    Without half-open a breaker either never recovers or recovers by sending the whole
    fleet at a provider that has just come back.
    """
    threshold: int = 3
    cooldown: float = 2.0
    failures: int = 0
    opened_at: float = 0.0
    state: str = "closed"
    trips: int = 0

    def __call__(self, fn: Callable, *args, **kwargs):
        if self.state == "open":
            if time.monotonic() - self.opened_at < self.cooldown:
                raise Open("circuit is open")
            self.state = "half-open"
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.failures += 1
            if self.state == "half-open" or self.failures >= self.threshold:
                self.state, self.opened_at = "open", time.monotonic()
                self.trips += 1
            raise
        self.failures, self.state = 0, "closed"
        return result


@dataclass
class Bulkhead:
    """
    A fixed number of concurrent calls to one dependency, and no more.

    The name is from ships: a hull divided into compartments so one breach does not
    sink the vessel. Here the compartment is a semaphore, and the breach is a
    dependency that has become slow rather than broken — which is worse, because
    nothing errors and every worker ends up waiting on it.
    """
    limit: int = 4
    rejected: int = 0
    _gate: threading.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self._gate = threading.Semaphore(self.limit)

    def __call__(self, fn: Callable, *args, **kwargs):
        if not self._gate.acquire(blocking=False):
            self.rejected += 1
            raise Open("bulkhead full")
        try:
            return fn(*args, **kwargs)
        finally:
            self._gate.release()
