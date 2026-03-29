"""
Feature engineering: cost prediction and metadata embedding.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class CostPredictor:
    """
    Model pricing table for coarse request-cost prediction.
    Prices are USD per 1 000 tokens.  New providers can be added via
    ``register_price()``.
    """

    def __init__(self) -> None:
        self.pricing_per_1k: Dict[str, float] = {
            "OpenAI-GPT4": 0.030,
            "Anthropic-Claude": 0.015,
            "Google-Gemini": 0.010,
            "Local-Llama3-8B": 0.002,
        }

    def register_price(self, model_name: str, price_per_1k: float) -> None:
        self.pricing_per_1k[model_name] = price_per_1k

    def estimate(self, token_count: int, model_name: str = "Google-Gemini") -> float:
        price = self.pricing_per_1k.get(model_name, self.pricing_per_1k["Google-Gemini"])
        return (max(1, token_count) / 1000.0) * price


class MetadataEmbeddingLayer:
    """Maps request metadata into normalised features for quantum encoding."""

    def __init__(self, max_tokens: int = 4096, max_expected_cost: float = 0.02) -> None:
        self.max_tokens = max_tokens
        self.max_expected_cost = max_expected_cost

    def build_vector(
        self,
        token_count: int,
        complexity_score: float,
        metadata: Dict[str, Any],
        estimated_cost: float,
    ) -> np.ndarray:
        priority = float(metadata.get("user_priority", 0.5))
        budget = float(
            metadata.get("model_constraints", {}).get(
                "max_budget_usd", self.max_expected_cost
            )
        )
        normalised_tokens = min(1.0, max(0.0, token_count / float(self.max_tokens)))
        normalised_complexity = min(1.0, max(0.0, complexity_score))
        priority_level = min(1.0, max(0.0, priority))
        budget_constraint = 1.0 - min(1.0, max(0.0, budget / self.max_expected_cost))

        return np.array(
            [normalised_tokens, normalised_complexity, priority_level, budget_constraint],
            dtype=float,
        )
