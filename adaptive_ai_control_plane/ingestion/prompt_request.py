"""Core data structures for the Nexus-Q AI Control Plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict
import time
import uuid


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VIP = "vip"

    @property
    def weight(self) -> int:
        return {"low": 1, "medium": 2, "high": 3, "vip": 4}[self.value]


@dataclass
class PromptRequest:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    budget_limit: float = 1.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    @property
    def is_vip(self) -> bool:
        return self.priority == Priority.VIP

    @property
    def is_high_priority(self) -> bool:
        return self.priority in (Priority.HIGH, Priority.VIP)


@dataclass
class RoutingDecision:
    provider: str
    twin_prediction: str
    pacemaker_status: str
    reasoning: str
    confidence: float = 0.0
    estimated_cost: float = 0.0
    estimated_latency: float = 0.0
    quantum_resolved: bool = False

    def explain(self, roi: float = 0.0) -> str:
        return (
            f"[Nexus-Q] Routed to {self.provider} "
            f"| Reason: {self.reasoning} "
            f"| Twin Prediction: {self.twin_prediction} "
            f"| ROI: ${roi:.4f}"
        )
