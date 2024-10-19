import os
import random
from enum import StrEnum

import torch
import numpy as np


class IntervalStrategy(StrEnum):
    EPOCH = "epoch"
    STEP = "step"
    NO = "no"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_full_determinism(seed: int, warn_only: bool = False):
    # set seed
    set_seed(seed)

    # Enable PyTorch deterministic mode. This potentially requires either the environment
    # variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # # The environment variable required to enable deterministic mode on Ascend NPUs.
    # os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    # os.environ["HCCL_DETERMINISTIC"] = "1"

    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    torch.use_deterministic_algorithms(True, warn_only=warn_only)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
