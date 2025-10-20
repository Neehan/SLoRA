from transformers import Trainer


class BaselineTrainer(Trainer):
    """Standard HF Trainer - uses model's internal loss computation."""
    pass
