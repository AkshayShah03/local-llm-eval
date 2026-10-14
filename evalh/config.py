from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    family: str
    params_b: float | None
    quant: str
    provider: str = "ollama"


@dataclass(frozen=True)
class Grid:
    defaults: dict
    models: list[ModelConfig]
    not_run: list[dict]


def load_grid(path: str | Path) -> Grid:
    raw = yaml.safe_load(Path(path).read_text())
    models = [ModelConfig(**m) for m in raw["models"]]
    return Grid(defaults=raw["defaults"], models=models, not_run=raw.get("not_run", []))


def load_acceptance(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())["use_cases"]
