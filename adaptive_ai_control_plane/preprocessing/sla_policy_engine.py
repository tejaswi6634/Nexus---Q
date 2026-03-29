"""SLA Policy Engine — enforces business rules that override or constrain routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from adaptive_ai_control_plane.ingestion.prompt_request import PromptRequest, Priority


@dataclass
class PolicyVerdict:
    forced_provider: Optional[str]
    blocked_providers: List[str]
    reason: str


class SLAPolicyEngine:
    """Evaluates a request against a hierarchy of business rules and returns
    hard constraints the RL agent must respect."""

    def evaluate(
        self,
        request: PromptRequest,
        intent: str,
        budget_remaining: float,
    ) -> PolicyVerdict:
        forced: Optional[str] = None
        blocked: List[str] = []
        reasons: List[str] = []

        if intent == "Legal":
            forced = "Local-Llama3"
            reasons.append("Legal-intent forces local model for data privacy")

        if request.is_vip and forced is None:
            blocked.extend(["Mistral-7B"])
            reasons.append("VIP traffic excluded from lowest-quality tier")

        if budget_remaining < 0.01:
            forced = "Mistral-7B"
            reasons.append("Budget exhausted — routing to cheapest provider")
        elif budget_remaining < 0.05:
            blocked.extend(["GPT-4"])
            reasons.append("Low budget — GPT-4 blocked")

        if request.priority == Priority.LOW:
            blocked.extend(["GPT-4"])
            reasons.append("Low-priority excluded from premium tier")

        reason_str = "; ".join(reasons) if reasons else "No policy override"
        return PolicyVerdict(
            forced_provider=forced,
            blocked_providers=list(set(blocked)),
            reason=reason_str,
        )
