from transformers import Trainer


class BaselineTrainer(Trainer):
    """Standard HF Trainer with no custom loss computation."""
    pass
