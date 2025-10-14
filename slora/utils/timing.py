import time
from typing import Optional


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        assert self.start_time is not None, "Timer not started"
        self.elapsed = time.perf_counter() - self.start_time
        if self.name:
            print(f"{self.name}: {self.elapsed:.4f}s")

    def reset(self):
        """Reset timer."""
        self.start_time = None
        self.elapsed = None
