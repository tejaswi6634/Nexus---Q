"""
Resilient Inference Engine — Core Data Models.

Defines the canonical request/response types and supporting enumerations
used throughout every layer of the resilience stack.  All routing
metadata (predictive-failure flags, confidence scores, fallback traces)
is captured in ``UnifiedInferenceResponse`` for full observability.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enumerations ─────────────────────────────────────────────────────

class UserTier(Enum):
    """SLA tier that drives fallback strategy selection."""
    PREMIUM = "premium"
    STANDARD = "standard"


class ProviderStatus(Enum):
    """Predictive health classification for a provider endpoint."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


# ── Health Metrics ───────────────────────────────────────────────────

@dataclass
class HealthMetrics:
    """Rolling-window health snapshot for a single provider."""
    provider: str
    latency_moving_avg: float = 0.0
    latency_baseline: float = 0.0
    latency_trend_pct: float = 0.0
    error_rate: float = 0.0
    timeout_frequency: float = 0.0
    sample_count: int = 0
    status: ProviderStatus = ProviderStatus.HEALTHY


# ── Digital Twin Simulation ──────────────────────────────────────────

@dataclass
class SimulationOutcome:
    """Predicted outcome produced by the Digital Twin before a fallback."""
    provider: str
    estimated_latency: float
    estimated_cost: float
    estimated_quality: float
    composite_score: float
    recommendation: str


# ── Unified Request / Response ───────────────────────────────────────

@dataclass
class UnifiedInferenceRequest:
    """
    Enriched request object flowing through the Resilient Inference Engine.

    Extends the base ``PromptRequest`` with SLA tier, complexity label,
    cost estimate, and a unique request identifier for trace correlation.
    """
    prompt_text: str
    user_tier: UserTier = UserTier.STANDARD
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    complexity_label: str = "medium"
    estimated_cost: float = 0.005
    max_latency_ms: float = 2000.0
    timestamp: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class UnifiedInferenceResponse:
    """
    Canonical response carrying both the LLM output and full
    observability metadata required for audit and RL feedback.

    Key metadata fields:
        was_predicted_failure  — True if the predictive health engine
                                 flagged a provider *before* it failed.
        confidence_score       — Post-response quality confidence ∈ [0, 1].
        fallback_trace         — Ordered list of routing decisions and
                                 layer activations during this request.
    """
    provider: str
    response_text: str
    latency: float
    cost: float
    quality: float
    success: bool

    # Observability metadata
    was_predicted_failure: bool
    confidence_score: float
    fallback_trace: List[str]

    was_cache_hit: bool = False
    was_compressed: bool = False
    compression_ratio: float = 1.0
    verification_provider: Optional[str] = None
    simulation_outcome: Optional[SimulationOutcome] = None
    request_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
