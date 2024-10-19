import shutil
import logging
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..utils import is_package_avaiable
from .callback import Callback
from ..trainer_state import TrainerState
from ..trainer_args import TrainerArguments

logger = logging.getLogger(__name__)
is_transformers_avaiable = is_package_avaiable("transformers")
is_peft_avaiable = is_package_avaiable("peft")

if is_transformers_avaiable:
    from transformers import PreTrainedModel
if is_peft_avaiable:
    from peft import PeftModel


class CheckpointCallback(Callback):
    BASE_DIR = "checkpoints"
    CKPT_DIR = "checkpoint-{global_step}"
    PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
    SAFETENSORS_WEIGHTS_NAME = "model.safetensors"

    def __init__(self, safe_serialization: bool = True, limit: int = 1):
        super().__init__()
        self.BASE_DIR = Path(self.BASE_DIR)
        self.BASE_DIR.mkdir(parents=True, exist_ok=True)
        self.safe_serialization = safe_serialization
        self.limit = limit

    def on_save(self, args: TrainerArguments, state: TrainerState) -> None:
        save_dir = self.BASE_DIR / self.CKPT_DIR.format(global_step=state.global_step)
        save_dir.mkdir(parents=True, exist_ok=True)

        # save model ckpt
        state_dict = self.model.state_dict()
        if (
            is_transformers_avaiable
            and isinstance(self.model, PreTrainedModel)
            or is_peft_avaiable
            and isinstance(self.model, PeftModel)
        ):
            self.accelerator.unwrap_model(self.model).save_pretrained(
                save_dir,
                state_dict=state_dict,
                safe_serialization=self.safe_serialization,
            )
        elif self.safe_serialization:
            save_file(state_dict, save_dir / self.SAFETENSORS_WEIGHTS_NAME)
        else:
            torch.save(state_dict, save_dir / self.PYTORCH_WEIGHTS_NAME)

        # limit
        if (
            sorted_checkpoints := sorted(
                [
                    x
                    for x in self.BASE_DIR.glob(self.CKPT_DIR.format(global_step="*"))
                    if x.is_dir()
                ],
                key=lambda path: path.stat().st_mtime,
            )
        ) is not None:
            if len(sorted_checkpoints) < self.limit:
                return

            # delete
            for checkpoint in sorted_checkpoints[
                : len(sorted_checkpoints) - self.limit
            ]:
                shutil.rmtree(checkpoint)
                logger.info(f"Deleting old checkpoint {checkpoint}")
