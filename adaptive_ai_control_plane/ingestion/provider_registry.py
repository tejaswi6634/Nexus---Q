"""Provider registry with telemetry tracking and simulated inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple
import random
import time


@dataclass
class ProviderProfile:
    name: str
    base_latency: float
    cost_per_1k_tokens: float
    quality_score: float
    is_local: bool = False
    rate_limit_rpm: int = 60
    request_count: int = 0
    _rate_window: deque = field(default_factory=lambda: deque(maxlen=200))
    latency_history: deque = field(default_factory=lambda: deque(maxlen=20))

    def record_request(self, latency: float) -> None:
        self.request_count += 1
        self._rate_window.append(time.time())
        self.latency_history.append(latency)

    def is_rate_limited(self) -> bool:
        now = time.time()
        recent = sum(1 for t in self._rate_window if now - t < 60)
        return recent >= self.rate_limit_rpm

    def avg_recent_latency(self, window: int = 5) -> Optional[float]:
        if len(self.latency_history) < window:
            return None
        recent = list(self.latency_history)[-window:]
        return sum(recent) / len(recent)

    def simulate_response(
        self, complexity: str, inject_degradation: bool = False
    ) -> Tuple[float, float, float, bool]:
        """Simulate a provider response.

        Returns (latency, cost, quality, rate_limited).
        """
        if self.is_rate_limited():
            return 0.0, 0.0, 0.0, True

        multiplier = {"Simple": 1.0, "Medium": 1.5, "Complex": 2.5}.get(complexity, 1.5)
        jitter = random.uniform(0.8, 1.4)
        degradation = random.uniform(1.3, 1.8) if inject_degradation else 1.0

        latency = self.base_latency * multiplier * jitter * degradation
        cost = self.cost_per_1k_tokens * multiplier * 0.5
        quality = self.quality_score * random.uniform(0.85, 1.0)

        return round(latency, 4), round(cost, 6), round(quality, 4), False


class ProviderRegistry:
    """Central registry of all available LLM providers."""

    def __init__(self) -> None:
        self.providers: Dict[str, ProviderProfile] = {
            "GPT-4": ProviderProfile(
                name="GPT-4",
                base_latency=1.2,
                cost_per_1k_tokens=0.030,
                quality_score=0.95,
                rate_limit_rpm=40,
            ),
            "Claude-3": ProviderProfile(
                name="Claude-3",
                base_latency=0.9,
                cost_per_1k_tokens=0.025,
                quality_score=0.92,
                rate_limit_rpm=50,
            ),
            "Gemini-1.5": ProviderProfile(
                name="Gemini-1.5",
                base_latency=0.35,
                cost_per_1k_tokens=0.010,
                quality_score=0.84,
                rate_limit_rpm=60,
            ),
            "Local-Llama3": ProviderProfile(
                name="Local-Llama3",
                base_latency=0.4,
                cost_per_1k_tokens=0.001,
                quality_score=0.78,
                is_local=True,
                rate_limit_rpm=200,
            ),
            "Mistral-7B": ProviderProfile(
                name="Mistral-7B",
                base_latency=0.3,
                cost_per_1k_tokens=0.002,
                quality_score=0.75,
                is_local=True,
                rate_limit_rpm=200,
            ),
        }

    def get(self, name: str) -> ProviderProfile:
        if name not in self.providers:
            raise KeyError(f"Unknown provider: {name}")
        return self.providers[name]

    def all_profiles(self) -> List[ProviderProfile]:
        return list(self.providers.values())

    def provider_names(self) -> List[str]:
        return list(self.providers.keys())

    def cheapest_local(self) -> str:
        local = [p for p in self.providers.values() if p.is_local]
        return min(local, key=lambda p: p.cost_per_1k_tokens).name

    def update_telemetry(self, provider_name: str, latency: float) -> None:
        self.get(provider_name).record_request(latency)

    def latency_history_snapshot(
        self, provider_names: List[str], n: int = 8
    ) -> Dict[str, List[float]]:
        """Last ``n`` recorded latencies per provider (for explainability UI)."""
        out: Dict[str, List[float]] = {}
        for name in provider_names:
            hist = list(self.get(name).latency_history)[-n:]
            out[name] = [float(x) for x in hist]
        return out

    def gpt4_equivalent_cost(self, complexity: str) -> float:
        """Return what GPT-4 would have cost for this complexity tier."""
        multiplier = {"Simple": 1.0, "Medium": 1.5, "Complex": 2.5}.get(complexity, 1.5)
        return self.providers["GPT-4"].cost_per_1k_tokens * multiplier * 0.5
