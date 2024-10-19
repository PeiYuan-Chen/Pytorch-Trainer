from .callback import Callback
from ..trainer_state import TrainerState
from ..trainer_args import TrainerArguments


class TensorBoardCallback(Callback):
    def __init__(self):
        try:
            import torch.utils.tensorboard
        except ImportError as ex:
            raise ImportError(
                "You want to use `TensorBoard` which is not installed yet, "
                "install it via `pip install tensorboard`."
            ) from ex
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter()

    def on_log(self, args: TrainerArguments, state: TrainerState, logs: dict) -> None:
        if self.accelerator.is_main_process:
            self.writer.add_scalars("", logs, state.global_step)
