"""
resilience — Advanced Resilient Inference Engine.

Provides predictive health monitoring, circuit breakers, confidence-based
quality gating, Digital Twin simulation, semantic caching, prompt
compression, and a multi-level local fallback hierarchy — all
orchestrated by the ``ResilientRouter``.
"""

from .models import (
    HealthMetrics,
    ProviderStatus,
    SimulationOutcome,
    UnifiedInferenceRequest,
    UnifiedInferenceResponse,
    UserTier,
)
from .health_engine import ProviderHealthMonitor
from .circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState
from .confidence_scorer import ConfidenceScorer
from .fallback_simulator import FallbackSimulator
from .semantic_cache import SemanticCache
from .prompt_compressor import CompressionResult, PromptCompressor
from .local_hierarchy import LocalFallbackHierarchy
from .strategies import (
    CostOptimizedStrategy,
    FallbackStrategy,
    LatencyOptimizedStrategy,
    QualityFirstStrategy,
    strategy_for_tier,
)
from .resilient_router import ResilientRouter

__all__ = [
    # Core orchestrator
    "ResilientRouter",
    # Models
    "HealthMetrics",
    "ProviderStatus",
    "SimulationOutcome",
    "UnifiedInferenceRequest",
    "UnifiedInferenceResponse",
    "UserTier",
    # Subsystems
    "ProviderHealthMonitor",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitState",
    "ConfidenceScorer",
    "FallbackSimulator",
    "SemanticCache",
    "CompressionResult",
    "PromptCompressor",
    "LocalFallbackHierarchy",
    # Strategy pattern
    "FallbackStrategy",
    "QualityFirstStrategy",
    "CostOptimizedStrategy",
    "LatencyOptimizedStrategy",
    "strategy_for_tier",
]
