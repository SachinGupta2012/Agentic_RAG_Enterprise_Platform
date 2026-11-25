"""Optional request-local I/O bounds, inactive for standalone Phase 2/3 calls."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from typing import Literal, TypedDict


class BudgetExceeded(RuntimeError):
    """No further I/O may start for this run."""


class RunBudget:
    def __init__(
        self,
        seconds: float,
        attempts: int,
        tokens: int,
        reads: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.clock = clock
        self.deadline = clock() + seconds
        self.attempt_limit, self.token_limit, self.read_limit = attempts, tokens, reads
        self.llm_attempts = self.tokens_reserved = self.reads = 0
        self.cancelled = False
        self._lock = threading.Lock()

    def remaining(self) -> float:
        remaining = self.deadline - self.clock()
        if self.cancelled or remaining <= 0:
            raise BudgetExceeded("deadline_exceeded")
        return remaining

    def reserve(self, kind: Literal["llm", "read"], tokens: int = 0) -> float:
        with self._lock:
            remaining = self.remaining()
            if kind == "llm":
                if (
                    self.llm_attempts >= self.attempt_limit
                    or self.tokens_reserved + tokens > self.token_limit
                ):
                    raise BudgetExceeded("llm_budget_exhausted")
                self.llm_attempts += 1
                self.tokens_reserved += tokens
            else:
                if self.reads >= self.read_limit:
                    raise BudgetExceeded("read_budget_exhausted")
                self.reads += 1
            return remaining

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True


active_budget: ContextVar[RunBudget | None] = ContextVar("support_run_budget", default=None)


class TimeoutKwargs(TypedDict, total=False):
    timeout: int


def read_timeout() -> TimeoutKwargs:
    budget = active_budget.get()
    if budget is None:
        return {}
    return {"timeout": max(1, int(budget.reserve("read")))}
