"""
RL-Centric Routing Brain & Decision Engine.

The RL agent is promoted to the **primary decision maker**.  Every routing
call follows this path:

  1. Build state from (Complexity label + Health Score + Provider Stats)
  2. RL ``act()`` returns: selected_provider, is_override_triggered,
     retry_strategy
  3. Post-response state synchronisation updates the agent **immediately**
     (rate-limit counters, actual latency) → eliminates stale-data routing.

The old ``if/else`` heuristic is gone; the only hard override remaining is
when *all* cloud providers are degraded (forced Local fallback).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .rl_agent import QLearningAgent
from providers.registry import ProviderRegistry
from providers.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDecision:
    """Structured output of a single routing decision."""
    selected_provider: str
    is_override_triggered: bool
    retry_strategy: str          # "none" | "immediate_fallback" | "exponential"
    reasoning: str


def _health_bucket(score: float) -> str:
    if score < 0.25:
        return "H_GOOD"
    if score < 0.50:
        return "H_FAIR"
    return "H_POOR"


def _cost_bucket(cost: float) -> str:
    if cost < 0.003:
        return "C_LOW"
    if cost < 0.008:
        return "C_MID"
    return "C_HIGH"


class DecisionEngine:
    """
    The BRAIN of the system.

    Consults the RL agent as the **primary** decision maker.  The agent
    observes (complexity_label, health_bucket, rate_limited, cost_bucket)
    and selects a provider action.  Hard overrides apply only when safety
    requires it (e.g. all cloud endpoints degraded).
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        health_monitor: HealthMonitor,
    ) -> None:
        self.registry = registry
        self.health = health_monitor
        self.action_map: Dict[int, str] = {
            idx: name for idx, name in enumerate(registry.provider_names)
        }
        self.agent = QLearningAgent(n_actions=len(self.action_map))

        self._last_state: Optional[Tuple[str, ...]] = None
        self._last_action: Optional[int] = None
        self._last_decision_ts: float = 0.0

    def route(
        self,
        complexity_label: str,
        estimated_cost: float,
        provider_rate_limited: bool = False,
    ) -> RoutingDecision:
        """
        Primary entry-point: ask the RL agent where to send this request.

        Returns
        -------
        RoutingDecision
            Contains ``selected_provider``, ``is_override_triggered``,
            ``retry_strategy``, and a human-readable ``reasoning`` string.
        """
        best_provider = self.health.best_provider() or "Local-Llama3-8B"
        best_health = self.health.score(best_provider)

        state = self.agent.get_state_key(
            complexity_label=complexity_label,
            health_bucket=_health_bucket(best_health),
            rate_limited=provider_rate_limited,
            cost_bucket=_cost_bucket(estimated_cost),
        )

        action_idx = self.agent.choose_action(state)
        chosen = self.action_map.get(action_idx, "Local-Llama3-8B")

        override = False
        retry_strategy = "none"

        prof = self.registry.get(chosen)
        if prof is not None and prof.is_rate_limited:
            override = True
            retry_strategy = "immediate_fallback"
            healthy = self.health.healthy_providers()
            chosen = min(healthy, key=healthy.get) if healthy else "Local-Llama3-8B"  # type: ignore[arg-type]

        if self.health.is_degraded(chosen):
            override = True
            retry_strategy = "exponential"
            healthy = self.health.healthy_providers()
            if healthy:
                chosen = min(healthy, key=healthy.get)  # type: ignore[arg-type]
            else:
                chosen = "Local-Llama3-8B"

        reasoning = (
            f"Routed to {chosen} | complexity={complexity_label}, "
            f"health={best_health:.2f}, cost_bucket={_cost_bucket(estimated_cost)}"
        )
        if override:
            reasoning += f" [OVERRIDE: {retry_strategy}]"
        logger.info(reasoning)

        self._last_state = state
        self._last_action = action_idx
        self._last_decision_ts = time.time()

        return RoutingDecision(
            selected_provider=chosen,
            is_override_triggered=override,
            retry_strategy=retry_strategy,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Post-Response State Synchronisation (Module 5)
    # ------------------------------------------------------------------
    def synchronise_state(
        self,
        provider: str,
        latency: float,
        quality: float,
        cost: float,
        success: bool,
        rate_limited: bool,
        *,
        reward: Optional[float] = None,
        next_complexity_label: str = "medium",
        next_estimated_cost: float = 0.005,
    ) -> float:
        """
        MUST be called **immediately** after a provider response is received.

        1. Pushes telemetry into the ``ProviderRegistry`` (rate-limit
           counters, actual latency, quality).
        2. Computes the RL reward and performs a Q-learning update so
           the agent never routes on stale data.

        Returns the computed reward.
        """
        self.registry.record_response(
            name=provider,
            latency=latency,
            quality=quality,
            success=success,
            rate_limited=rate_limited,
        )

        if reward is None:
            reward = self.compute_reward(quality, latency, cost)

        if self._last_state is not None and self._last_action is not None:
            best_health = self.health.score(
                self.health.best_provider() or "Local-Llama3-8B"
            )
            next_state = self.agent.get_state_key(
                complexity_label=next_complexity_label,
                health_bucket=_health_bucket(best_health),
                rate_limited=rate_limited,
                cost_bucket=_cost_bucket(next_estimated_cost),
            )
            self.agent.learn(self._last_state, self._last_action, reward, next_state)

        return reward

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------
    @staticmethod
    def compute_reward(
        quality: float,
        latency: float,
        cost: float,
        *,
        alpha: float = 1.0,
        beta: float = 0.6,
        gamma: float = 120.0,
    ) -> float:
        return float(alpha * quality - beta * latency - gamma * cost)

    # ------------------------------------------------------------------
    # Adaptive parameter tuning (kept for backwards compat)
    # ------------------------------------------------------------------
    def adjust_parameters(
        self, current_rate: float, current_dim: int
    ) -> Tuple[float, int]:
        best_score = self.health.score(
            self.health.best_provider() or "Local-Llama3-8B"
        )
        if best_score > 0.5:
            new_rate = max(1.0, current_rate * 0.9)
            new_dim = max(2, current_dim - 1)
        elif best_score < 0.2:
            new_rate = min(50.0, current_rate * 1.1)
            new_dim = min(4, current_dim + 1)
        else:
            new_rate, new_dim = current_rate, current_dim
        return new_rate, new_dim
