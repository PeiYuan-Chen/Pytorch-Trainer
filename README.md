# Pytorch-Trainer

## Usage
train.py
```
import hydra
from omegaconf import DictConfig
from src.trainer import Trainer


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    with Trainer.from_hydra_config(cfg) as trainer:
        trainer.train()


if __name__ == "__main__":
    main()
```

`accelerate launch train.py`