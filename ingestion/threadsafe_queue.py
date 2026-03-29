from queue import Queue, Empty, Full
from typing import Optional

from .prompt_request import PromptRequest


class ThreadSafeQueue:
    """
    Thin typed wrapper over Queue for PromptRequest transport.
    """

    def __init__(self, maxsize: int = 100):
        self._queue: Queue[PromptRequest] = Queue(maxsize=maxsize)

    def put(self, item: PromptRequest, timeout: Optional[float] = None) -> None:
        self._queue.put(item, timeout=timeout)

    def get(self, timeout: Optional[float] = None) -> PromptRequest:
        return self._queue.get(timeout=timeout)

    def qsize(self) -> int:
        return self._queue.qsize()

    @staticmethod
    def empty_exception():
        return Empty

    @staticmethod
    def full_exception():
        return Full
