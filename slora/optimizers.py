import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Any, Callable


class GatedOptimizer:
    """
    Wraps optimizer to conditionally skip weight updates on rejected batches.

    Prevents weight decay and other per-step optimizer updates when batch is rejected.
    Handles gradient accumulation: accepts if ANY batch in window was accepted.
    """

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self._should_step = False
        self.num_accepted_in_window = 0

    def mark_accept(self) -> None:
        self._should_step = True
        self.num_accepted_in_window += 1

    def get_num_accepted(self) -> int:
        return self.num_accepted_in_window

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if self._should_step:
            result = self.optimizer.step(closure)
            self._should_step = False
            self.num_accepted_in_window = 0
            return result
        self._should_step = False
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)
        self.num_accepted_in_window = 0

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.optimizer, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("optimizer", "_should_step", "num_accepted_in_window"):
            super().__setattr__(name, value)
        else:
            setattr(self.optimizer, name, value)


class GatedLRScheduler:
    """
    Wraps LR scheduler to conditionally skip scheduling on rejected batches.

    Prevents learning rate updates when batch is rejected.
    Handles gradient accumulation: accepts if ANY batch in window was accepted.
    """

    def __init__(self, scheduler: LRScheduler):
        self.scheduler = scheduler
        self._should_step = False

    def mark_accept(self) -> None:
        self._should_step = True

    def step(self, *args: Any, **kwargs: Any) -> None:
        if self._should_step:
            self.scheduler.step(*args, **kwargs)
            self._should_step = False
        else:
            self._should_step = False

    def state_dict(self) -> dict[str, Any]:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict)

    def get_last_lr(self) -> list[float]:
        return self.scheduler.get_last_lr()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.scheduler, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("scheduler", "_should_step"):
            super().__setattr__(name, value)
        else:
            setattr(self.scheduler, name, value)
