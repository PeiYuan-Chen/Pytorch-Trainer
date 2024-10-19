from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from ..trainer_state import TrainerState
from ..trainer_args import TrainerArguments


class Callback:
    def __init__(self):
        self.accelerator: Accelerator | None = None
        self.model: nn.Module | None = None
        self.train_loader: DataLoader | None = None
        self.eval_loader: DataLoader | None = None

    def setup(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader

    def on_train_init(self, args: TrainerArguments, state: TrainerState) -> None:
        pass

    def on_train_begin(self, args: TrainerArguments, state: TrainerState) -> None:
        pass

    def on_train_end(self, args: TrainerArguments, state: TrainerState) -> None:
        pass

    def on_epoch_begin(self, args: TrainerArguments, state: TrainerState) -> None:
        pass

    def on_epoch_end(self, args: TrainerArguments, state: TrainerState) -> None:
        pass

    def on_step_end(self, args: TrainerArguments, state: TrainerState) -> None:
        pass

    def on_eval_begin(self, args: TrainerArguments, state: TrainerState) -> None:
        pass

    def on_eval_step_end(self, args: TrainerArguments, state: TrainerState) -> None:
        pass

    def on_eval_end(self, args: TrainerArguments, state: TrainerState) -> None:
        pass

    def on_log(self, args: TrainerArguments, state: TrainerState, logs: dict) -> None:
        pass

    def on_save(self, args: TrainerArguments, state: TrainerState) -> None:
        pass
