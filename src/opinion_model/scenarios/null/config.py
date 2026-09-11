"""Configuration schema and TOML loading for the null scenario."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import tomllib

from opinion_model.shared import SimulationConfig


def _positive(value: float, name: str) -> float:
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


@dataclass(frozen=True)
class NullInitializationConfig:
    """Sparse-network and heterogeneous ordinary-belief initialization."""

    network_m: int
    ordinary_mean_alpha: float
    ordinary_concentration: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.network_m, bool)
            or not isinstance(self.network_m, int)
            or self.network_m < 1
        ):
            raise ValueError("network_m must be a positive integer.")
        _positive(self.ordinary_mean_alpha, "ordinary_mean_alpha")
        _positive(self.ordinary_concentration, "ordinary_concentration")


@dataclass(frozen=True)
class NullConfig:
    """Resolved shared and initialization values for one null run."""

    simulation: SimulationConfig
    initialization: NullInitializationConfig

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, SimulationConfig):
            raise TypeError("simulation must be a SimulationConfig.")
        if not isinstance(self.initialization, NullInitializationConfig):
            raise TypeError(
                "initialization must be a NullInitializationConfig."
            )
        if self.initialization.network_m >= self.simulation.agent_count:
            raise ValueError("network_m must be below agent_count.")


@dataclass(frozen=True)
class NullExperimentConfig:
    """Matched-run definition for the exploratory null scenario."""

    experiment_id: str
    status: str
    seeds: tuple[int, ...]
    extremism_threshold: float
    null: NullConfig

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must be non-empty.")
        if not self.status:
            raise ValueError("status must be non-empty.")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique.")
        for seed in self.seeds:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("Every seed must be a non-negative integer.")
        threshold = float(self.extremism_threshold)
        if not isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError(
                "extremism_threshold must lie strictly between 0 and 1."
            )
        if not isinstance(self.null, NullConfig):
            raise TypeError("null must be a NullConfig.")


def load_null_experiment_config(path: str | Path) -> NullExperimentConfig:
    """Load one null experiment from a TOML file."""
    source = Path(path)
    with source.open("rb") as stream:
        raw = tomllib.load(stream)

    forbidden_sections = {"opinion_leader", "platform"} & set(raw)
    if forbidden_sections:
        raise ValueError(
            "Null configuration cannot contain focal mechanism sections: "
            f"{sorted(forbidden_sections)}"
        )

    experiment = raw["experiment"]
    simulation = raw["simulation"]
    initialization = raw["initialization"]
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    null = NullConfig(
        simulation=SimulationConfig(
            agent_count=int(simulation["agent_count"]),
            rounds=int(simulation["rounds"]),
            seed=seeds[0],
            initial_mean=0.5,
            initial_concentration=float(
                initialization["ordinary_concentration"]
            ),
            base_origination_probability=float(
                simulation["base_origination_probability"]
            ),
            evidence_weight=float(simulation["evidence_weight"]),
            consumption_capacity=int(simulation["consumption_capacity"]),
            exclude_self_messages=bool(simulation["exclude_self_messages"]),
        ),
        initialization=NullInitializationConfig(
            network_m=int(initialization["network_m"]),
            ordinary_mean_alpha=float(initialization["ordinary_mean_alpha"]),
            ordinary_concentration=float(
                initialization["ordinary_concentration"]
            ),
        ),
    )
    return NullExperimentConfig(
        experiment_id=str(experiment["id"]),
        status=str(experiment["status"]),
        seeds=seeds,
        extremism_threshold=float(experiment["extremism_threshold"]),
        null=null,
    )


__all__ = [
    "NullConfig",
    "NullExperimentConfig",
    "NullInitializationConfig",
    "load_null_experiment_config",
]
