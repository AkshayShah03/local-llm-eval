from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Backend:
    name: str
    provider: str
    model: str


@dataclass(frozen=True)
class Route:
    name: str
    data_class: str
    chain: list[str]
    first_token_timeout_s: float
    total_timeout_s: float


@dataclass(frozen=True)
class GatewayConfig:
    backends: dict[str, Backend]
    routes: dict[str, Route]


def validate(cfg: GatewayConfig) -> GatewayConfig:
    for route in cfg.routes.values():
        for name in route.chain:
            if name not in cfg.backends:
                raise ConfigError(f"route {route.name}: unknown backend {name}")
            if route.data_class == "internal" and cfg.backends[name].provider != "ollama":
                raise ConfigError(f"route {route.name} is internal but lists external backend {name}")
    return cfg


def load_config(path: str | Path) -> GatewayConfig:
    raw = yaml.safe_load(Path(path).read_text())
    backends = {n: Backend(n, b["provider"], b["model"]) for n, b in raw["backends"].items()}
    routes = {n: Route(n, r["data_class"], r["chain"], r["first_token_timeout_s"], r["total_timeout_s"])
              for n, r in raw["routes"].items()}
    return validate(GatewayConfig(backends, routes))
