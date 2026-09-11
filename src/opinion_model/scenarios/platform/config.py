"""Configuration schema and TOML loading for the platform-only scenario."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import tomllib

from opinion_model.platform import PlatformMessageSelection, PlatformNetworkUpdate
from opinion_model.shared import SimulationConfig


def _positive(value: float, name: str) -> float:
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


@dataclass(frozen=True)
class PlatformInitializationConfig:
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
class PlatformMechanismConfig:
    """Parameters for reach, attention, and exposure-driven network adaptation."""

    out_of_network_availability_probability: float
    formation_midpoint_probability: float
    formation_degree_log_odds_strength: float
    formation_alignment_log_odds_strength: float
    dissolution_midpoint_probability: float
    dissolution_degree_log_odds_strength: float
    dissolution_alignment_log_odds_strength: float

    def __post_init__(self) -> None:
        self.build_message_selection()
        self.build_network_update()

    def build_message_selection(self) -> PlatformMessageSelection:
        return PlatformMessageSelection(
            self.out_of_network_availability_probability,
        )

    def build_network_update(self) -> PlatformNetworkUpdate:
        return PlatformNetworkUpdate(
            formation_midpoint_probability=self.formation_midpoint_probability,
            formation_degree_log_odds_strength=(
                self.formation_degree_log_odds_strength
            ),
            formation_alignment_log_odds_strength=(
                self.formation_alignment_log_odds_strength
            ),
            dissolution_midpoint_probability=self.dissolution_midpoint_probability,
            dissolution_degree_log_odds_strength=(
                self.dissolution_degree_log_odds_strength
            ),
            dissolution_alignment_log_odds_strength=(
                self.dissolution_alignment_log_odds_strength
            ),
        )


@dataclass(frozen=True)
class PlatformConfig:
    """Resolved shared, initialization, and platform values for one run."""

    simulation: SimulationConfig
    initialization: PlatformInitializationConfig
    platform: PlatformMechanismConfig

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, SimulationConfig):
            raise TypeError("simulation must be a SimulationConfig.")
        if not isinstance(self.initialization, PlatformInitializationConfig):
            raise TypeError(
                "initialization must be a PlatformInitializationConfig."
            )
        if self.initialization.network_m >= self.simulation.agent_count:
            raise ValueError("network_m must be below agent_count.")
        if not isinstance(self.platform, PlatformMechanismConfig):
            raise TypeError("platform must be a PlatformMechanismConfig.")


@dataclass(frozen=True)
class PlatformExperimentConfig:
    """Matched-run definition for the exploratory platform scenario."""

    experiment_id: str
    status: str
    seeds: tuple[int, ...]
    extremism_threshold: float
    platform_case: PlatformConfig

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
        if not isinstance(self.platform_case, PlatformConfig):
            raise TypeError("platform_case must be a PlatformConfig.")


def load_platform_experiment_config(
    path: str | Path,
) -> PlatformExperimentConfig:
    """Load one platform-only experiment from a TOML file."""
    source = Path(path)
    with source.open("rb") as stream:
        raw = tomllib.load(stream)

    if "opinion_leader" in raw:
        raise ValueError(
            "Platform-only configuration cannot contain an opinion_leader section."
        )

    experiment = raw["experiment"]
    simulation = raw["simulation"]
    initialization = raw["initialization"]
    platform = raw["platform"]
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    platform_case = PlatformConfig(
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
        initialization=PlatformInitializationConfig(
            network_m=int(initialization["network_m"]),
            ordinary_mean_alpha=float(initialization["ordinary_mean_alpha"]),
            ordinary_concentration=float(
                initialization["ordinary_concentration"]
            ),
        ),
        platform=PlatformMechanismConfig(
            out_of_network_availability_probability=float(
                platform["out_of_network_availability_probability"]
            ),
            formation_midpoint_probability=float(
                platform["formation_midpoint_probability"]
            ),
            formation_degree_log_odds_strength=float(
                platform["formation_degree_log_odds_strength"]
            ),
            formation_alignment_log_odds_strength=float(
                platform["formation_alignment_log_odds_strength"]
            ),
            dissolution_midpoint_probability=float(
                platform["dissolution_midpoint_probability"]
            ),
            dissolution_degree_log_odds_strength=float(
                platform["dissolution_degree_log_odds_strength"]
            ),
            dissolution_alignment_log_odds_strength=float(
                platform["dissolution_alignment_log_odds_strength"]
            ),
        ),
    )
    return PlatformExperimentConfig(
        experiment_id=str(experiment["id"]),
        status=str(experiment["status"]),
        seeds=seeds,
        extremism_threshold=float(experiment["extremism_threshold"]),
        platform_case=platform_case,
    )


__all__ = [
    "PlatformConfig",
    "PlatformExperimentConfig",
    "PlatformInitializationConfig",
    "PlatformMechanismConfig",
    "load_platform_experiment_config",
]
