from dataclasses import dataclass, field
from typing import Dict, Any
import time


@dataclass
class PromptRequest:
    """
    LLM-centric request object used throughout the routing pipeline.
    """

    prompt_text: str
    metadata: Dict[str, Any]
    token_count: int
    timestamp: float = field(default_factory=time.time)
