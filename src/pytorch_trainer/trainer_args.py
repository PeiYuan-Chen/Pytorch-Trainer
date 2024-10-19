from pathlib import Path
from dataclasses import dataclass

from .trainer_utils import IntervalStrategy


@dataclass
class TrainerArguments:
    epochs: int = 1
    # deterministic
    seed: int = 42
    full_determinism: bool = False
    # log
    log_strategy: str | IntervalStrategy = "epoch"
    log_interval: int = 1
    # evaluate
    eval_strategy: str | IntervalStrategy = "epoch"
    eval_interval: int = 1
    # save
    save_strategy: str | IntervalStrategy = "epoch"
    save_interval: int = 1
    safe_serialization: bool = True
    # gradient
    max_grad_norm: float = 1.0
    # monitor
    monitor: str = "eval_loss"

    def __post_init__(self):
        # log
        self.log_strategy = IntervalStrategy(self.log_strategy)
        # validation
        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        # save
        self.save_strategy = IntervalStrategy(self.save_strategy)
        # monitor
        if not self.monitor.startswith("eval_"):
            self.monitor = f"eval_{self.monitor}"
