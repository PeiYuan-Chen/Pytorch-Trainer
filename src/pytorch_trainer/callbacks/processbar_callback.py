import math

from tqdm.auto import tqdm

from .callback import Callback
from ..trainer_state import TrainerState
from ..trainer_args import TrainerArguments


class ProcessBarCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_epoch_bar = None
        self.train_step_bar = None
        self.eval_bar = None

    def on_train_begin(self, args: TrainerArguments, state: TrainerState) -> None:
        if self.accelerator.is_main_process:
            self.train_epoch_bar = tqdm(
                total=args.epochs - state.epoch,
                desc="Train Epochs",
                leave=False,
                dynamic_ncols=True,
            )

    def on_epoch_begin(
        self,
        args: TrainerArguments,
        state: TrainerState,
    ) -> None:
        if self.accelerator.is_main_process:
            self.train_step_bar = tqdm(
                total=math.ceil(
                    len(self.train_loader)
                    / self.accelerator.gradient_accumulation_steps
                ),
                desc="Train Steps",
                leave=False,
                dynamic_ncols=True,
            )

    def on_step_end(self, args: TrainerArguments, state: TrainerState) -> None:
        if self.accelerator.is_main_process:
            self.train_step_bar.update(1)

    def on_epoch_end(self, args: TrainerArguments, state: TrainerState) -> None:
        if self.accelerator.is_main_process:
            self.train_step_bar.close()
            self.train_step_bar = None
            self.train_epoch_bar.update(1)

    def on_eval_begin(self, args: TrainerArguments, state: TrainerState) -> None:
        if self.accelerator.is_main_process:
            self.eval_bar = tqdm(
                total=len(self.eval_loader),
                desc="Evaluation",
                leave=False,
                dynamic_ncols=True,
            )

    def on_eval_step_end(self, args: TrainerArguments, state: TrainerState) -> None:
        if self.accelerator.is_main_process:
            self.eval_bar.update(1)

    def on_eval_end(self, args: TrainerArguments, state: TrainerState) -> None:
        if self.accelerator.is_main_process:
            self.eval_bar.close()
            self.eval_bar = None
