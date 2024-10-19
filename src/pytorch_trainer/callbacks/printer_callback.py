from .callback import Callback
from ..trainer_state import TrainerState
from ..trainer_args import TrainerArguments


class PrinterCallback(Callback):
    def on_log(self, args: TrainerArguments, state: TrainerState, logs: dict) -> None:
        logs["epoch"] = state.epoch
        logs["global_step"] = state.global_step
        self.accelerator.print(logs)
