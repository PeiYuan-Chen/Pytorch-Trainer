from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0

    def save_to_json(self, filepath: str | Path):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f)

    def load_from_json(self, filepath: str | Path):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
