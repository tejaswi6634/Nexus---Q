"""
Circuit Breaker — prevents cascading failure death-spirals.

Each provider gets its own breaker with three states:

    CLOSED    — Normal operation.  Consecutive failures are counted.
    OPEN      — Provider is failing.  Requests are rejected immediately
                to avoid wasting latency budget and amplifying load on
                a struggling endpoint.
    HALF_OPEN — After ``recovery_timeout`` elapses, a small number of
                probe requests are allowed.  If they succeed the breaker
                resets to CLOSED; if they fail it re-opens.

``CircuitBreakerRegistry`` is the factory / manager that lazily creates
and caches per-provider breakers.

Thread-safe for concurrent routing threads.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Dict, List

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-provider circuit breaker with configurable thresholds."""

    def __init__(
        self,
        provider: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 2,
    ) -> None:
        self.provider = provider
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and time.time() - self._last_failure_time >= self._recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._success_count = 0
                logger.info(
                    "CircuitBreaker[%s] OPEN → HALF_OPEN", self.provider
                )
            return self._state

    def can_execute(self) -> bool:
        st = self.state
        if st == CircuitState.CLOSED:
            return True
        if st == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self._half_open_max_calls:
                    self._half_open_calls += 1
                    return True
            return False
        return False

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._half_open_max_calls:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(
                        "CircuitBreaker[%s] HALF_OPEN → CLOSED (recovered)",
                        self.provider,
                    )
            else:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker[%s] HALF_OPEN → OPEN (still failing)",
                    self.provider,
                )
            elif self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker[%s] CLOSED → OPEN (failures=%d)",
                    self.provider, self._failure_count,
                )

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0


class CircuitBreakerRegistry:
    """Lazily creates and caches per-provider ``CircuitBreaker`` instances."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._lock = threading.Lock()

    def get(self, provider: str) -> CircuitBreaker:
        with self._lock:
            if provider not in self._breakers:
                self._breakers[provider] = CircuitBreaker(
                    provider=provider,
                    failure_threshold=self._failure_threshold,
                    recovery_timeout=self._recovery_timeout,
                )
            return self._breakers[provider]

    def all_executable(self) -> List[str]:
        with self._lock:
            return [p for p, cb in self._breakers.items() if cb.can_execute()]

    @property
    def breakers(self) -> Dict[str, CircuitBreaker]:
        return dict(self._breakers)
