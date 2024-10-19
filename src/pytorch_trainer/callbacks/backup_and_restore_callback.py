import math
import logging
from pathlib import Path

from .callback import Callback
from ..trainer_state import TrainerState
from ..trainer_args import TrainerArguments

logger = logging.getLogger(__name__)


class BackupAndResumeCallback(Callback):
    BACKUP_DIR = "backup"
    TRAINER_STATE_PATH = "trainer_state.json"

    def __init__(self, safe_serialization: bool = True):
        super().__init__()
        self.BACKUP_DIR = Path(self.BACKUP_DIR)
        self.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        self.safe_serialization = safe_serialization

    def on_train_init(self, args: TrainerArguments, state: TrainerState):
        # resume training
        if (
            any(self.BACKUP_DIR.iterdir())
            and (self.BACKUP_DIR / self.TRAINER_STATE_PATH).exists()
        ):
            self.accelerator.load_state(self.BACKUP_DIR)
            state.load_from_json(self.BACKUP_DIR / self.TRAINER_STATE_PATH)

            skip_batches = (
                state.global_step
                % (
                    math.ceil(
                        len(self.train_loader)
                        / self.accelerator.gradient_accumulation_steps
                    )
                )
                * self.accelerator.gradient_accumulation_steps
            )
            self.train_loader = self.accelerator.skip_first_batches(
                self.train_loader, skip_batches
            )

            logger.info(f"Resuming Training from checkpoint: {self.BACKUP_DIR}")
            logger.info(
                f"Resuming Training from epoch {state.epoch}, global_step {state.global_step}"
            )
            logger.info(
                f"Will Skip the first {skip_batches} batches in the first epoch."
            )
        else:
            logger.info("No checkpoint found. Starting Training from scratch.")

    def on_save(self, args: TrainerArguments, state: TrainerState):
        # save checkpoint
        self.accelerator.save_state(self.BACKUP_DIR, self.safe_serialization)
        state.save_to_json(self.BACKUP_DIR / self.TRAINER_STATE_PATH)
