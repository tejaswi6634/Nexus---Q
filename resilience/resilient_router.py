"""
Advanced Resilient Inference Engine — ``ResilientRouter``.

Orchestrates every request through five defence layers:

    ┌──────────────────────────────────────────────────────────────┐
    │  1. PREDICTIVE HEALTH ENGINE  (pre-emptive layer)           │
    │     → Detect degrading providers BEFORE they fail            │
    ├──────────────────────────────────────────────────────────────┤
    │  2. GRACEFUL DEGRADATION  (efficiency layer)                │
    │     → Compress long prompts when global latency is high      │
    ├──────────────────────────────────────────────────────────────┤
    │  3. SEMANTIC CACHE  (survival layer — level 3)              │
    │     → Return cached response if semantically identical       │
    ├──────────────────────────────────────────────────────────────┤
    │  4. PRIMARY PROVIDER via RL + Circuit Breaker                │
    │     → RL routing engine ➜ circuit-breaker gate ➜ call        │
    ├──────────────────────────────────────────────────────────────┤
    │  5. CONFIDENCE GATE + DIGITAL TWIN FALLBACK  (quality +     │
    │     intelligence layers)                                     │
    │     → Score response confidence; if low, simulate fallback   │
    │       outcomes via the Digital Twin and re-route              │
    ├──────────────────────────────────────────────────────────────┤
    │  6. LOCAL FALLBACK HIERARCHY  (survival layer)              │
    │     → Llama-3-8B → Phi-3-Mini → SemanticCache               │
    └──────────────────────────────────────────────────────────────┘

Every response is wrapped in a ``UnifiedInferenceResponse`` that
includes:
    * ``was_predicted_failure``  — pre-emptive detection flag
    * ``confidence_score``       — post-response quality metric
    * ``fallback_trace``         — ordered list of every routing hop
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

from providers.registry import ProviderRegistry
from providers.health_monitor import HealthMonitor
from orchestration.decision_engine import DecisionEngine

from .models import (
    HealthMetrics,
    ProviderStatus,
    SimulationOutcome,
    UnifiedInferenceRequest,
    UnifiedInferenceResponse,
    UserTier,
)
from .health_engine import ProviderHealthMonitor
from .circuit_breaker import CircuitBreakerRegistry
from .confidence_scorer import ConfidenceScorer
from .fallback_simulator import FallbackSimulator
from .semantic_cache import SemanticCache
from .prompt_compressor import PromptCompressor
from .local_hierarchy import LocalFallbackHierarchy
from .strategies import FallbackStrategy, strategy_for_tier

logger = logging.getLogger(__name__)


class ResilientRouter:
    """
    Top-level entry point for the resilient inference pipeline.

    Parameters
    ----------
    registry : ProviderRegistry
        Live provider telemetry store.
    health_monitor : HealthMonitor
        Legacy health scorer (kept for RL state construction).
    decision_engine : DecisionEngine
        RL-based routing brain.
    provider_call_fn : callable
        ``(provider_name: str, base_cost: float) -> dict``
        Function that executes (or simulates) a provider call.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        health_monitor: HealthMonitor,
        decision_engine: DecisionEngine,
        provider_call_fn: Callable,
        *,
        health_window_seconds: float = 60.0,
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: float = 30.0,
        confidence_threshold: float = 0.40,
        cache_similarity_threshold: float = 0.82,
        high_latency_threshold: float = 1.5,
    ) -> None:
        self.registry = registry
        self.health_monitor = health_monitor
        self.decision_engine = decision_engine
        self._call_fn = provider_call_fn

        # Layer 1 — Predictive Health
        self.predictive_health = ProviderHealthMonitor(
            window_seconds=health_window_seconds,
        )
        for name in registry.provider_names:
            prof = registry.get(name)
            self.predictive_health.register_provider(
                name, prof.avg_latency if prof else 0.5,
            )
        self.predictive_health.register_provider("Local-Phi3-Mini", 0.12)

        # Circuit Breakers
        self.circuit_breakers = CircuitBreakerRegistry(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_recovery_timeout,
        )

        # Layer 2 — Digital Twin
        self.fallback_simulator = FallbackSimulator()

        # Layer 3 — Confidence Scorer
        self.confidence_scorer = ConfidenceScorer(
            verification_threshold=confidence_threshold,
        )

        # Layer 4 — Semantic Cache & Local Hierarchy
        self.semantic_cache = SemanticCache(
            similarity_threshold=cache_similarity_threshold,
        )
        self.local_hierarchy = LocalFallbackHierarchy(
            semantic_cache=self.semantic_cache,
            provider_call_fn=provider_call_fn,
        )

        # Layer 5 — Prompt Compressor
        self.prompt_compressor = PromptCompressor()

        self._high_latency_threshold = high_latency_threshold
        self._global_latency_avg = 0.0
        self._latency_ema_alpha = 0.1
        self._total_requests = 0

    # ══════════════════════════════════════════════════════════════════
    #  PRIMARY INFERENCE PATH
    # ══════════════════════════════════════════════════════════════════

    def infer(self, request: UnifiedInferenceRequest) -> UnifiedInferenceResponse:
        """
        Route a single request through the full resilient pipeline.

        Returns a ``UnifiedInferenceResponse`` carrying the LLM output
        plus complete routing metadata for observability and RL feedback.
        """
        t0 = time.time()
        trace: List[str] = []
        was_predicted_failure = False
        was_compressed = False
        compression_ratio = 1.0
        working_prompt = request.prompt_text

        # ── 1. Predictive Health Check ───────────────────────────────
        degraded = self.predictive_health.get_degraded_providers()
        all_health = self.predictive_health.get_all_metrics()
        if degraded:
            was_predicted_failure = True
            trace.append(f"predictive_degraded:{','.join(degraded)}")

        # ── 2. Graceful Degradation (Prompt Compression) ─────────────
        if self.prompt_compressor.should_compress(
            working_prompt,
            self._global_latency_avg,
            self._high_latency_threshold,
        ):
            cr = self.prompt_compressor.compress(working_prompt)
            working_prompt = cr.compressed_text
            was_compressed = True
            compression_ratio = cr.compression_ratio
            trace.append(f"prompt_compressed:ratio={compression_ratio:.2f}")

        # ── 3. Semantic Cache Lookup ─────────────────────────────────
        cache_hit = self.semantic_cache.lookup(working_prompt)
        if cache_hit is not None:
            resp_text, sim, orig_prov = cache_hit
            trace.append(f"cache_hit:sim={sim:.3f},orig={orig_prov}")
            return UnifiedInferenceResponse(
                provider=f"SemanticCache(orig={orig_prov})",
                response_text=resp_text,
                latency=time.time() - t0,
                cost=0.0,
                quality=sim * 0.80,
                success=True,
                was_predicted_failure=was_predicted_failure,
                confidence_score=sim,
                fallback_trace=trace,
                was_cache_hit=True,
                was_compressed=was_compressed,
                compression_ratio=compression_ratio,
                request_id=request.request_id,
            )

        # ── 4. Primary Provider Selection (RL + health) ──────────────
        any_rl = any(
            p.is_rate_limited for p in self.registry.all_profiles().values()
        )
        decision = self.decision_engine.route(
            complexity_label=request.complexity_label,
            estimated_cost=request.estimated_cost,
            provider_rate_limited=any_rl,
        )
        selected = decision.selected_provider
        trace.append(f"rl_selected:{selected}")

        # Pre-emptive reroute away from predicted-degraded provider
        if selected in degraded:
            healthy_candidates = [
                p for p in self.registry.provider_names if p not in degraded
            ]
            if healthy_candidates:
                strategy = strategy_for_tier(request.user_tier)
                alt = strategy.select(
                    request, healthy_candidates, all_health,
                    self._profile_dict(healthy_candidates),
                )
                if alt:
                    selected = alt
                    trace.append(f"predictive_reroute:{selected}")

        # ── 5. Circuit Breaker Gate ──────────────────────────────────
        cb = self.circuit_breakers.get(selected)
        if not cb.can_execute():
            trace.append(f"circuit_open:{selected}")
            available = [
                p for p in self.registry.provider_names
                if p not in degraded
                and self.circuit_breakers.get(p).can_execute()
            ]
            if available:
                strategy = strategy_for_tier(request.user_tier)
                alt = strategy.select(
                    request, available, all_health,
                    self._profile_dict(available),
                )
                selected = alt or available[0]
                trace.append(f"circuit_reroute:{selected}")
            else:
                trace.append("circuit_all_open→local_hierarchy")
                return self._local_fallback(
                    request, working_prompt, trace, t0,
                    True, was_compressed, compression_ratio,
                )

        # ── 6. Provider Call ─────────────────────────────────────────
        result = self._call_fn(selected, request.estimated_cost)
        trace.append(f"called:{selected}")

        if result.get("rate_limited") and not result.get("success", True):
            self._record_failure(selected, result)
            trace.append(f"rate_limited:{selected}")
            return self._local_fallback(
                request, working_prompt, trace, t0,
                True, was_compressed, compression_ratio,
            )

        if not result.get("success", False):
            self._record_failure(selected, result)
            trace.append(f"call_failed:{selected}")
            return self._local_fallback(
                request, working_prompt, trace, t0,
                True, was_compressed, compression_ratio,
            )

        # Success → update health telemetry
        self.circuit_breakers.get(selected).record_success()
        self.predictive_health.record_event(selected, result["latency"])
        self.fallback_simulator.record_outcome(
            selected, result["latency"], result["cost"], result["quality"],
        )

        # ── 7. Confidence Evaluation ─────────────────────────────────
        confidence, needs_verify = self.confidence_scorer.evaluate(
            result["response_text"],
            expected_length_hint=max(20, request.token_count // 2),
        )
        trace.append(f"confidence:{confidence:.3f}")

        verification_provider: Optional[str] = None
        sim_outcome: Optional[SimulationOutcome] = None

        if needs_verify:
            trace.append("quality_fallback_triggered")
            v_result, verification_provider, sim_outcome = self._quality_fallback(
                request, selected, degraded, all_health, trace,
            )
            if v_result is not None:
                result = v_result
                confidence = self.confidence_scorer.score(result["response_text"])
                trace.append(
                    f"verified:{verification_provider},confidence:{confidence:.3f}"
                )

        # ── 8. Cache the successful response ─────────────────────────
        self.semantic_cache.store(
            request.prompt_text, result["response_text"],
            provider=selected, quality=result.get("quality", 0.5),
        )

        # ── 9. Update globals and return ─────────────────────────────
        elapsed = time.time() - t0
        total_latency = result["latency"] + elapsed * 0.1
        self._update_global_latency(total_latency)
        self._total_requests += 1

        return UnifiedInferenceResponse(
            provider=result.get("provider", selected),
            response_text=result["response_text"],
            latency=total_latency,
            cost=result["cost"],
            quality=result["quality"],
            success=True,
            was_predicted_failure=was_predicted_failure,
            confidence_score=confidence,
            fallback_trace=trace,
            was_cache_hit=False,
            was_compressed=was_compressed,
            compression_ratio=compression_ratio,
            verification_provider=verification_provider,
            simulation_outcome=sim_outcome,
            request_id=request.request_id,
        )

    # ══════════════════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _quality_fallback(
        self,
        request: UnifiedInferenceRequest,
        primary_provider: str,
        degraded: List[str],
        all_health: Dict[str, HealthMetrics],
        trace: List[str],
    ) -> Tuple[Optional[dict], Optional[str], Optional[SimulationOutcome]]:
        """Simulate outcomes via Digital Twin then call verifier model."""
        candidates = [
            p for p in self.registry.provider_names
            if p != primary_provider
            and p not in degraded
            and self.circuit_breakers.get(p).can_execute()
        ]
        if not candidates:
            return None, None, None

        ranked = self.fallback_simulator.rank_fallbacks(request, candidates)
        sim_outcome = ranked[0] if ranked else None
        verifier = sim_outcome.provider if sim_outcome else candidates[0]

        if sim_outcome:
            trace.append(
                f"twin_sim:{verifier}(score={sim_outcome.composite_score:.3f})"
            )

        try:
            v_result = self._call_fn(verifier, request.estimated_cost)
            if v_result.get("success", False):
                self.predictive_health.record_event(
                    verifier, v_result["latency"],
                )
                self.circuit_breakers.get(verifier).record_success()
                self.fallback_simulator.record_outcome(
                    verifier, v_result["latency"],
                    v_result["cost"], v_result["quality"],
                )
                return v_result, verifier, sim_outcome
        except Exception as exc:
            logger.warning("Quality-fallback to %s failed: %s", verifier, exc)

        return None, None, sim_outcome

    def _local_fallback(
        self,
        request: UnifiedInferenceRequest,
        prompt: str,
        trace: List[str],
        t0: float,
        was_predicted: bool,
        was_compressed: bool,
        compression_ratio: float,
    ) -> UnifiedInferenceResponse:
        """Cascade through the local hierarchy (Survival Layer)."""
        trace.append("local_hierarchy_engaged")
        result, level = self.local_hierarchy.execute(prompt, request.estimated_cost)
        elapsed = time.time() - t0

        if result is not None:
            trace.append(f"local_success:{level}")
            total_latency = result["latency"] + elapsed * 0.1
            self._update_global_latency(total_latency)

            prov = result.get("provider", level)
            if "SemanticCache" not in prov:
                self.predictive_health.record_event(prov, result["latency"])
                self.semantic_cache.store(
                    request.prompt_text, result["response_text"],
                    provider=prov, quality=result.get("quality", 0.5),
                )

            confidence = self.confidence_scorer.score(result["response_text"])
            return UnifiedInferenceResponse(
                provider=prov,
                response_text=result["response_text"],
                latency=total_latency,
                cost=result.get("cost", 0.0),
                quality=result.get("quality", 0.30),
                success=True,
                was_predicted_failure=was_predicted,
                confidence_score=confidence,
                fallback_trace=trace,
                was_compressed=was_compressed,
                compression_ratio=compression_ratio,
                request_id=request.request_id,
            )

        trace.append("total_failure")
        return UnifiedInferenceResponse(
            provider="NONE",
            response_text=(
                "Service temporarily unavailable. "
                "All providers and caches exhausted."
            ),
            latency=time.time() - t0,
            cost=0.0,
            quality=0.0,
            success=False,
            was_predicted_failure=was_predicted,
            confidence_score=0.0,
            fallback_trace=trace,
            was_compressed=was_compressed,
            compression_ratio=compression_ratio,
            request_id=request.request_id,
        )

    def _record_failure(self, provider: str, result: dict) -> None:
        self.circuit_breakers.get(provider).record_failure()
        self.predictive_health.record_event(
            provider, result.get("latency", 1.0),
            is_error=True, is_timeout=False,
        )

    def _profile_dict(self, providers: List[str]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for p in providers:
            prof = self.registry.get(p)
            if prof:
                out[p] = {"quality": prof.quality_trend}
        return out

    def _update_global_latency(self, latency: float) -> None:
        if self._global_latency_avg == 0.0:
            self._global_latency_avg = latency
        else:
            a = self._latency_ema_alpha
            self._global_latency_avg = (1 - a) * self._global_latency_avg + a * latency

    # ══════════════════════════════════════════════════════════════════
    #  STATE SYNC (delegates to underlying DecisionEngine)
    # ══════════════════════════════════════════════════════════════════

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
        """Push RL feedback into the underlying ``DecisionEngine``."""
        return self.decision_engine.synchronise_state(
            provider=provider,
            latency=latency,
            quality=quality,
            cost=cost,
            success=success,
            rate_limited=rate_limited,
            reward=reward,
            next_complexity_label=next_complexity_label,
            next_estimated_cost=next_estimated_cost,
        )

    # ══════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ══════════════════════════════════════════════════════════════════

    def start_health_monitoring(self, interval: float = 5.0) -> None:
        """Launch the background predictive-health monitor thread."""
        self.predictive_health.start_background_monitor(interval)

    def stop_health_monitoring(self) -> None:
        self.predictive_health.stop_background_monitor()

    # ══════════════════════════════════════════════════════════════════
    #  DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════

    def get_diagnostics(self) -> dict:
        """Snapshot of all engine subsystem states for debugging / UI."""
        return {
            "total_requests": self._total_requests,
            "global_latency_avg": round(self._global_latency_avg, 5),
            "cache_size": self.semantic_cache.size,
            "health_metrics": {
                p: {
                    "status": m.status.value,
                    "latency_avg": round(m.latency_moving_avg, 4),
                    "error_rate": round(m.error_rate, 4),
                    "latency_trend_pct": round(m.latency_trend_pct, 4),
                }
                for p, m in self.predictive_health.get_all_metrics().items()
            },
            "circuit_breakers": {
                p: cb.state.value
                for p, cb in self.circuit_breakers.breakers.items()
            },
        }
