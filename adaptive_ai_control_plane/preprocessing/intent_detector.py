"""Intent detection for incoming prompts — Technical, Legal, Creative, or General."""

from __future__ import annotations

from typing import Dict, Tuple
import re


_INTENT_LEXICONS: Dict[str, list] = {
    "Technical": [
        "algorithm", "code", "function", "api", "database", "architecture",
        "programming", "debug", "deploy", "server", "kubernetes", "docker",
        "microservice", "pipeline", "sql", "python", "java", "rust",
        "compiler", "binary", "git", "repository", "framework", "library",
        "optimize", "latency", "throughput", "benchmark", "cache", "index",
        "query", "schema", "migration", "endpoint", "http", "tcp", "udp",
        "socket", "thread", "process", "async", "await", "concurrency",
        "implement", "refactor", "infrastructure", "devops", "ci/cd",
        "machine learning", "neural network", "model", "training",
        "inference", "tensor", "gradient", "backpropagation", "gpu",
    ],
    "Legal": [
        "compliance", "regulation", "gdpr", "privacy", "contract", "legal",
        "law", "liability", "intellectual property", "patent", "copyright",
        "trademark", "license", "lawsuit", "litigation", "clause",
        "indemnification", "tort", "jurisdiction", "statute", "regulation",
        "hipaa", "sox", "pci", "ferpa", "ccpa", "data protection",
        "confidentiality", "non-disclosure", "nda", "arbitration",
        "terms of service", "policy", "audit", "sanctions", "aml", "kyc",
    ],
    "Creative": [
        "story", "poem", "creative", "write a", "imagine", "fiction",
        "narrative", "character", "plot", "dialogue", "novel", "essay",
        "brainstorm", "ideate", "artistic", "metaphor", "lyric", "song",
        "design a logo", "illustration", "visual", "slogan", "tagline",
        "brand", "copywriting", "script", "screenplay", "fantasy",
    ],
}

_COMPILED_PATTERNS: Dict[str, re.Pattern] = {
    intent: re.compile(
        r"\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\b",
        re.IGNORECASE,
    )
    for intent, keywords in _INTENT_LEXICONS.items()
}


class IntentDetector:
    """Keyword-density intent classifier."""

    INTENTS = ["Technical", "Legal", "Creative", "General"]

    def detect(self, text: str) -> str:
        scores = self._score(text)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "General"

    def detect_with_scores(self, text: str) -> Tuple[str, Dict[str, float]]:
        scores = self._score(text)
        best = max(scores, key=scores.get)
        intent = best if scores[best] > 0 else "General"
        return intent, scores

    def intent_index(self, text: str) -> int:
        return self.INTENTS.index(self.detect(text))

    def _score(self, text: str) -> Dict[str, float]:
        word_count = max(len(text.split()), 1)
        scores: Dict[str, float] = {}
        for intent, pattern in _COMPILED_PATTERNS.items():
            matches = pattern.findall(text)
            scores[intent] = len(matches) / word_count
        return scores
