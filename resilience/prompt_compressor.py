"""
Prompt Compressor — the 'Efficiency' Layer (Graceful Degradation).

When the engine detects sustained high global latency, large prompts
are compressed before dispatch to fallback (often smaller) models.

Compression phases:
    1. **Boilerplate stripping** — removes common preamble phrases
       ("you are a helpful assistant", "think step by step", …).
    2. **Structural pruning** — strips ``System:``, ``Background:``,
       ``Instructions:`` blocks that are non-essential to the query.
    3. **Whitespace normalisation** — collapses redundant newlines/spaces.
    4. **Truncation** — if the text is still above the target, it is
       trimmed to the most-significant leading tokens.

The original prompt intent is preserved; only padding is removed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

_STRIP_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(
        r"(?:^|\n)(?:System|Note|Context|Background):\s*.*?(?=\n[A-Z]|\Z)",
        re.DOTALL | re.I,
    ), ""),
    (re.compile(r"\n{3,}"), "\n\n"),
    (re.compile(r"[ \t]{2,}"), " "),
    (re.compile(
        r"(?:^|\n)#+\s*Instructions?\s*\n.*?(?=\n#+|\Z)",
        re.DOTALL | re.I,
    ), ""),
]

_BOILERPLATE_PHRASES: List[str] = [
    "please provide a detailed and comprehensive",
    "take your time to think through",
    "you are an expert in",
    "you are a helpful assistant",
    "as a senior engineer",
    "think step by step",
    "let's work through this",
    "here is some background context",
    "for additional context",
    "keep in mind that",
]


@dataclass
class CompressionResult:
    """Outcome of a prompt compression pass."""
    compressed_text: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    tokens_saved_estimate: int


class PromptCompressor:
    """
    Strips non-essential prompt context to reduce compute load on
    fallback models during high-latency conditions.

    Parameters
    ----------
    target_reduction : float
        Desired fraction of content to remove (0.40 = keep ≈ 60 %).
    min_output_words : int
        Safety floor — never compress below this many words.
    """

    def __init__(
        self,
        target_reduction: float = 0.40,
        min_output_words: int = 10,
    ) -> None:
        self._target_reduction = target_reduction
        self._min_output_words = min_output_words

    def compress(self, prompt: str) -> CompressionResult:
        original_len = len(prompt)
        working = prompt

        for phrase in _BOILERPLATE_PHRASES:
            working = re.sub(re.escape(phrase), "", working, flags=re.I)

        for pattern, repl in _STRIP_PATTERNS:
            working = pattern.sub(repl, working)

        working = re.sub(r"\s+", " ", working).strip()

        target_words = max(
            self._min_output_words,
            int(len(prompt.split()) * (1.0 - self._target_reduction)),
        )
        words = working.split()
        if len(words) > target_words:
            working = " ".join(words[:target_words])

        compressed_len = len(working)
        ratio = compressed_len / original_len if original_len > 0 else 1.0

        return CompressionResult(
            compressed_text=working,
            original_length=original_len,
            compressed_length=compressed_len,
            compression_ratio=ratio,
            tokens_saved_estimate=max(0, (original_len - compressed_len) // 4),
        )

    def should_compress(
        self,
        prompt: str,
        avg_latency: float,
        threshold_latency: float = 1.5,
    ) -> bool:
        """True when conditions warrant compression (high latency + long prompt)."""
        return avg_latency > threshold_latency and len(prompt.split()) > 50
