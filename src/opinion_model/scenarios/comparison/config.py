"""Load and validate the canonical four-scenario comparison contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import tomllib

from opinion_model.scenarios.baseline import (
    BaselineExperimentConfig,
    load_baseline_experiment_config,
)
from opinion_model.scenarios.null import (
    NullExperimentConfig,
    load_null_experiment_config,
)
from opinion_model.scenarios.opleader import (
    OpleaderExperimentConfig,
    load_opleader_experiment_config,
)
from opinion_model.scenarios.platform import (
    PlatformExperimentConfig,
    load_platform_experiment_config,
)
from opinion_model.shared import SimulationConfig


SCENARIO_NAMES = ("null", "opleader", "platform", "baseline")


def _shared_simulation_signature(config: SimulationConfig) -> tuple[object, ...]:
    return (
        config.agent_count,
        config.rounds,
        config.initial_mean,
        config.initial_concentration,
        config.base_origination_probability,
        config.interest_decay,
        config.evidence_weight,
        config.exclude_self_messages,
    )


def _ordinary_initialization_signature(config: object) -> tuple[object, ...]:
    return (
        getattr(config, "network_m"),
        getattr(config, "ordinary_mean_alpha"),
        getattr(config, "ordinary_concentration"),
    )


def _leader_initialization_signature(config: object) -> tuple[object, ...]:
    return (
        getattr(config, "leader_share"),
        getattr(config, "leader_positive_a"),
        getattr(config, "leader_positive_b"),
    )


def _opinion_leader_signature(config: object) -> tuple[object, ...]:
    return (
        getattr(config, "leader_log_odds_advantage"),
        getattr(config, "leader_evidence_multiplier"),
    )


def _platform_signature(config: object) -> tuple[object, ...]:
    names = (
        "out_of_network_availability_probability",
        "formation_midpoint_probability",
        "formation_degree_log_odds_strength",
        "formation_alignment_log_odds_strength",
        "dissolution_midpoint_probability",
        "dissolution_degree_log_odds_strength",
        "dissolution_alignment_log_odds_strength",
    )
    return tuple(getattr(config, name) for name in names)


@dataclass(frozen=True)
class ComparisonExperimentConfig:
    """One validated 2x2 comparison assembled from four scenario configs."""

    experiment_id: str
    status: str
    null: NullExperimentConfig
    opleader: OpleaderExperimentConfig
    platform: PlatformExperimentConfig
    baseline: BaselineExperimentConfig
    source_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("comparison id must be non-empty.")
        if not self.status:
            raise ValueError("comparison status must be non-empty.")
        expected_types = {
            "null": (self.null, NullExperimentConfig),
            "opleader": (self.opleader, OpleaderExperimentConfig),
            "platform": (self.platform, PlatformExperimentConfig),
            "baseline": (self.baseline, BaselineExperimentConfig),
        }
        for name, (value, expected_type) in expected_types.items():
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} has the wrong experiment-config type.")

        scenario_ids = {
            self.null.experiment_id,
            self.opleader.experiment_id,
            self.platform.experiment_id,
            self.baseline.experiment_id,
        }
        if scenario_ids != set(SCENARIO_NAMES):
            raise ValueError(
                "Scenario experiment ids must be null, opleader, platform, and baseline."
            )

        scenario_statuses = {
            self.null.status,
            self.opleader.status,
            self.platform.status,
            self.baseline.status,
        }
        if scenario_statuses != {self.status}:
            raise ValueError(
                "Every scenario status must match the comparison status."
            )

        seeds = {
            self.null.seeds,
            self.opleader.seeds,
            self.platform.seeds,
            self.baseline.seeds,
        }
        if len(seeds) != 1:
            raise ValueError("All comparison scenarios must use identical seeds.")

        thresholds = {
            self.null.extremism_threshold,
            self.opleader.extremism_threshold,
            self.platform.extremism_threshold,
            self.baseline.extremism_threshold,
        }
        if len(thresholds) != 1:
            raise ValueError(
                "All comparison scenarios must use the same extremism threshold."
            )

        simulations = (
            self.null.null.simulation,
            self.opleader.opleader.simulation,
            self.platform.platform_case.simulation,
            self.baseline.baseline.simulation,
        )
        signatures = {_shared_simulation_signature(value) for value in simulations}
        if len(signatures) != 1:
            raise ValueError(
                "Shared simulation parameters must match across all scenarios."
            )

        initializations = (
            self.null.null.initialization,
            self.opleader.opleader.initialization,
            self.platform.platform_case.initialization,
            self.baseline.baseline.initialization,
        )
        ordinary_signatures = {
            _ordinary_initialization_signature(value) for value in initializations
        }
        if len(ordinary_signatures) != 1:
            raise ValueError(
                "Network and ordinary-belief initialization must match across scenarios."
            )

        if _leader_initialization_signature(
            self.opleader.opleader.initialization
        ) != _leader_initialization_signature(self.baseline.baseline.initialization):
            raise ValueError(
                "Leader initialization must match between opleader and baseline."
            )
        if self.opleader.orientations != self.baseline.orientations:
            raise ValueError(
                "Leader orientations must match between opleader and baseline."
            )
        if _opinion_leader_signature(
            self.opleader.opleader.opinion_leader
        ) != _opinion_leader_signature(self.baseline.baseline.opinion_leader):
            raise ValueError(
                "Opinion-leader parameters must match between opleader and baseline."
            )
        if _platform_signature(
            self.platform.platform_case.platform
        ) != _platform_signature(self.baseline.baseline.platform):
            raise ValueError(
                "Platform parameters must match between platform and baseline."
            )

        no_platform_capacity = simulations[0].agent_count - 1
        if simulations[0].consumption_capacity != no_platform_capacity:
            raise ValueError("Null capacity must be nonbinding.")
        if simulations[1].consumption_capacity != no_platform_capacity:
            raise ValueError("Opleader capacity must be nonbinding.")
        if simulations[2].consumption_capacity != simulations[3].consumption_capacity:
            raise ValueError(
                "Platform and baseline must use the same finite attention capacity."
            )

        normalized_paths = {
            name: Path(path).resolve() for name, path in self.source_paths.items()
        }
        if set(normalized_paths) != set(SCENARIO_NAMES):
            raise ValueError("source_paths must contain every comparison scenario.")
        object.__setattr__(
            self,
            "source_paths",
            MappingProxyType(normalized_paths),
        )

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.null.seeds

    @property
    def extremism_threshold(self) -> float:
        return self.null.extremism_threshold


def load_comparison_experiment_config(
    path: str | Path,
) -> ComparisonExperimentConfig:
    """Load four scenario configs and reject any unmatched comparison input."""
    source = Path(path).resolve()
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    comparison = raw["comparison"]
    source_paths = {
        name: (source.parent / str(comparison[name])).resolve()
        for name in SCENARIO_NAMES
    }
    return ComparisonExperimentConfig(
        experiment_id=str(comparison["id"]),
        status=str(comparison["status"]),
        null=load_null_experiment_config(source_paths["null"]),
        opleader=load_opleader_experiment_config(source_paths["opleader"]),
        platform=load_platform_experiment_config(source_paths["platform"]),
        baseline=load_baseline_experiment_config(source_paths["baseline"]),
        source_paths=source_paths,
    )


__all__ = [
    "ComparisonExperimentConfig",
    "SCENARIO_NAMES",
    "load_comparison_experiment_config",
]
