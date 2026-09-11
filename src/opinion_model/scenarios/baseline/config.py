"""Configuration schema and TOML loading for the integrated baseline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal, TypeAlias, cast
import tomllib

from opinion_model.opleader import (
    OpinionLeaderMessageAggregation,
    OpinionLeaderMessageOrigination,
)
from opinion_model.platform import PlatformMessageSelection, PlatformNetworkUpdate
from opinion_model.shared import SimulationConfig


BaselineOrientation: TypeAlias = Literal[
    "positive",
    "negative",
    "balanced_positive",
    "balanced_negative",
]

BASELINE_ORIENTATIONS: tuple[BaselineOrientation, ...] = (
    "positive",
    "negative",
    "balanced_positive",
    "balanced_negative",
)


def _positive(value: float, name: str) -> float:
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return normalized


@dataclass(frozen=True)
class BaselineInitializationConfig:
    """Sparse-network and heterogeneous-belief initialization values."""

    network_m: int
    leader_share: float
    ordinary_mean_alpha: float
    ordinary_concentration: float
    leader_positive_a: float
    leader_positive_b: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.network_m, bool)
            or not isinstance(self.network_m, int)
            or self.network_m < 1
        ):
            raise ValueError("network_m must be a positive integer.")
        share = float(self.leader_share)
        if not isfinite(share) or not 0.0 < share < 1.0:
            raise ValueError("leader_share must lie strictly between 0 and 1.")
        _positive(self.ordinary_mean_alpha, "ordinary_mean_alpha")
        _positive(self.ordinary_concentration, "ordinary_concentration")
        positive_a = _positive(self.leader_positive_a, "leader_positive_a")
        positive_b = _positive(self.leader_positive_b, "leader_positive_b")
        if positive_a <= positive_b:
            raise ValueError(
                "leader_positive_a must exceed leader_positive_b."
            )

    def leader_count(self, agent_count: int) -> int:
        count = int(agent_count * self.leader_share + 0.5)
        if count < 1 or count >= agent_count:
            raise ValueError(
                "leader_share must select at least one but not every agent."
            )
        return count


@dataclass(frozen=True)
class OpinionLeaderMechanismConfig:
    """Parameters of the two opinion-leader advantages used by baseline."""

    leader_log_odds_advantage: float
    leader_evidence_multiplier: float

    def __post_init__(self) -> None:
        self.build_message_origination(frozenset())
        self.build_message_aggregation(frozenset())

    def build_message_origination(
        self,
        leader_ids: frozenset[int],
    ) -> OpinionLeaderMessageOrigination:
        return OpinionLeaderMessageOrigination(
            leader_ids=leader_ids,
            leader_log_odds_advantage=self.leader_log_odds_advantage,
        )

    def build_message_aggregation(
        self,
        leader_ids: frozenset[int],
    ) -> OpinionLeaderMessageAggregation:
        return OpinionLeaderMessageAggregation(
            leader_ids=leader_ids,
            leader_evidence_multiplier=self.leader_evidence_multiplier,
        )


@dataclass(frozen=True)
class PlatformMechanismConfig:
    """Parameters of platform reach and exposure-driven network adaptation."""

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
class BaselineConfig:
    """Resolved initialization, scheduler, and mechanism values for one run."""

    simulation: SimulationConfig
    initialization: BaselineInitializationConfig
    opinion_leader: OpinionLeaderMechanismConfig
    platform: PlatformMechanismConfig

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, SimulationConfig):
            raise TypeError("simulation must be a SimulationConfig.")
        if not isinstance(self.initialization, BaselineInitializationConfig):
            raise TypeError(
                "initialization must be a BaselineInitializationConfig."
            )
        if self.initialization.network_m >= self.simulation.agent_count:
            raise ValueError("network_m must be below agent_count.")
        self.initialization.leader_count(self.simulation.agent_count)
        if not isinstance(self.opinion_leader, OpinionLeaderMechanismConfig):
            raise TypeError(
                "opinion_leader must be an OpinionLeaderMechanismConfig."
            )
        if not isinstance(self.platform, PlatformMechanismConfig):
            raise TypeError("platform must be a PlatformMechanismConfig.")


@dataclass(frozen=True)
class BaselineExperimentConfig:
    """Matched-run definition for the exploratory integrated baseline."""

    experiment_id: str
    status: str
    seeds: tuple[int, ...]
    orientations: tuple[BaselineOrientation, ...]
    extremism_threshold: float
    baseline: BaselineConfig

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
        if not self.orientations or len(self.orientations) != len(
            set(self.orientations)
        ):
            raise ValueError("orientations must be non-empty and unique.")
        unknown = set(self.orientations) - set(BASELINE_ORIENTATIONS)
        if unknown:
            raise ValueError(f"Unknown baseline orientations: {sorted(unknown)}")
        threshold = float(self.extremism_threshold)
        if not isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("extremism_threshold must lie strictly between 0 and 1.")


def load_baseline_experiment_config(
    path: str | Path,
) -> BaselineExperimentConfig:
    """Load one baseline experiment from a TOML file."""
    source = Path(path)
    with source.open("rb") as stream:
        raw = tomllib.load(stream)

    experiment = raw["experiment"]
    simulation = raw["simulation"]
    initialization = raw["initialization"]
    opinion_leader = raw["opinion_leader"]
    platform = raw["platform"]

    seeds = tuple(int(seed) for seed in experiment["seeds"])
    orientations = tuple(
        cast(BaselineOrientation, orientation)
        for orientation in experiment["orientations"]
    )
    baseline = BaselineConfig(
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
            interest_decay=float(simulation["interest_decay"]),
            evidence_weight=float(simulation["evidence_weight"]),
            consumption_capacity=int(simulation["consumption_capacity"]),
            exclude_self_messages=bool(simulation["exclude_self_messages"]),
        ),
        initialization=BaselineInitializationConfig(
            network_m=int(initialization["network_m"]),
            leader_share=float(initialization["leader_share"]),
            ordinary_mean_alpha=float(initialization["ordinary_mean_alpha"]),
            ordinary_concentration=float(
                initialization["ordinary_concentration"]
            ),
            leader_positive_a=float(initialization["leader_positive_a"]),
            leader_positive_b=float(initialization["leader_positive_b"]),
        ),
        opinion_leader=OpinionLeaderMechanismConfig(
            leader_log_odds_advantage=float(
                opinion_leader["leader_log_odds_advantage"]
            ),
            leader_evidence_multiplier=float(
                opinion_leader["leader_evidence_multiplier"]
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
    return BaselineExperimentConfig(
        experiment_id=str(experiment["id"]),
        status=str(experiment["status"]),
        seeds=seeds,
        orientations=orientations,
        extremism_threshold=float(experiment["extremism_threshold"]),
        baseline=baseline,
    )


__all__ = [
    "BASELINE_ORIENTATIONS",
    "BaselineConfig",
    "BaselineExperimentConfig",
    "BaselineInitializationConfig",
    "BaselineOrientation",
    "OpinionLeaderMechanismConfig",
    "PlatformMechanismConfig",
    "load_baseline_experiment_config",
]
