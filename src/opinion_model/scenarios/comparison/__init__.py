"""Validated execution of the complete four-scenario comparison."""

from opinion_model.scenarios.comparison.config import (
    ComparisonExperimentConfig,
    SCENARIO_NAMES,
    load_comparison_experiment_config,
)
from opinion_model.scenarios.comparison.experiment import (
    COMMON_METRICS,
    SCENARIO_FACTORS,
    ComparisonResult,
    run_comparison_experiment,
)

__all__ = [
    "COMMON_METRICS",
    "ComparisonExperimentConfig",
    "ComparisonResult",
    "SCENARIO_FACTORS",
    "SCENARIO_NAMES",
    "load_comparison_experiment_config",
    "run_comparison_experiment",
]
