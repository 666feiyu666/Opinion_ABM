"""Execute and align the four matched OLIM 2.0 scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from opinion_model.scenarios.baseline.experiment import (
    final_condition_metrics as baseline_final_metrics,
    run_baseline_experiment,
)
from opinion_model.scenarios.comparison.config import ComparisonExperimentConfig
from opinion_model.scenarios.null.experiment import (
    final_condition_metrics as null_final_metrics,
    run_null_experiment,
)
from opinion_model.scenarios.opleader.experiment import (
    final_condition_metrics as opleader_final_metrics,
    run_opleader_experiment,
)
from opinion_model.scenarios.platform.experiment import (
    final_condition_metrics as platform_final_metrics,
    run_platform_experiment,
)


COMMON_METRICS = (
    "mean_signed_belief",
    "mean_absolute_belief",
    "extremist_ratio",
    "cumulative_content_balance",
    "edge_count",
    "mean_following_degree",
    "homophily_ratio",
)

SCENARIO_FACTORS = {
    "null": (False, False),
    "opleader": (True, False),
    "platform": (False, True),
    "baseline": (True, True),
}


@dataclass(frozen=True)
class ComparisonResult:
    """Aligned scenario outputs plus common final summaries."""

    scenario_round_metrics: dict[str, pd.DataFrame]
    round_metrics: pd.DataFrame
    final_metrics: pd.DataFrame
    final_summary: pd.DataFrame


def _annotate(
    frame: pd.DataFrame,
    scenario: str,
    *,
    final: bool = False,
) -> pd.DataFrame:
    result = frame.copy()
    leader_enabled, platform_enabled = SCENARIO_FACTORS[scenario]
    result.insert(0, "scenario", scenario)
    result.insert(1, "opinion_leader_enabled", leader_enabled)
    result.insert(2, "platform_enabled", platform_enabled)
    if "orientation" not in result:
        result.insert(3, "orientation", "none")
    if "condition" not in result:
        result.insert(4, "condition", result["orientation"])
    if not final:
        result["condition"] = result["condition"].fillna(result["orientation"])
    return result


def _summarize(final_metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "scenario",
        "opinion_leader_enabled",
        "platform_enabled",
        "condition",
    ]
    summary = final_metrics.groupby(group_columns)[list(COMMON_METRICS)].agg(
        ["mean", "std"]
    )
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    counts = final_metrics.groupby(group_columns).size().rename("run_count")
    return counts.to_frame().join(summary).reset_index()


def run_comparison_experiment(
    experiment: ComparisonExperimentConfig,
) -> ComparisonResult:
    """Run the matched 2x2 scenario set through their explicit assemblies."""
    if not isinstance(experiment, ComparisonExperimentConfig):
        raise TypeError("experiment must be a ComparisonExperimentConfig.")

    raw_rounds = {
        "null": run_null_experiment(experiment.null),
        "opleader": run_opleader_experiment(experiment.opleader),
        "platform": run_platform_experiment(experiment.platform),
        "baseline": run_baseline_experiment(experiment.baseline),
    }
    scenario_rounds = {
        name: _annotate(frame, name) for name, frame in raw_rounds.items()
    }
    round_metrics = pd.concat(
        scenario_rounds.values(),
        ignore_index=True,
        sort=False,
    )

    raw_final = {
        "null": null_final_metrics(raw_rounds["null"]),
        "opleader": opleader_final_metrics(raw_rounds["opleader"]),
        "platform": platform_final_metrics(raw_rounds["platform"]),
        "baseline": baseline_final_metrics(raw_rounds["baseline"]),
    }
    final_metrics = pd.concat(
        [
            _annotate(frame, name, final=True)
            for name, frame in raw_final.items()
        ],
        ignore_index=True,
        sort=False,
    )
    return ComparisonResult(
        scenario_round_metrics=scenario_rounds,
        round_metrics=round_metrics,
        final_metrics=final_metrics,
        final_summary=_summarize(final_metrics),
    )


__all__ = [
    "COMMON_METRICS",
    "ComparisonResult",
    "SCENARIO_FACTORS",
    "run_comparison_experiment",
]
