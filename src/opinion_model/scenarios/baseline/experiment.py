"""Matched execution of the exploratory integrated baseline."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from opinion_model.scenarios.baseline.assembly import assemble_baseline_components
from opinion_model.scenarios.baseline.config import (
    BaselineConfig,
    BaselineExperimentConfig,
    BaselineOrientation,
)
from opinion_model.scenarios.baseline.initialization import (
    BaselineInitialization,
    FixedBaselineInitializer,
    initialize_baseline,
)
from opinion_model.scenarios.baseline.observation import baseline_round_metrics
from opinion_model.shared import RandomStreams, SimulationResult, run_simulation


@dataclass(frozen=True)
class BaselineRunResult:
    """One resolved initialization, simulation result, and diagnostic table."""

    initialization: BaselineInitialization
    simulation_result: SimulationResult
    round_metrics: pd.DataFrame


def run_baseline_condition(
    config: BaselineConfig,
    orientation: BaselineOrientation,
    *,
    extremism_threshold: float,
) -> BaselineRunResult:
    """Run one orientation from its seed-matched initialized world."""
    random_streams = RandomStreams(config.simulation.seed)
    initialization = initialize_baseline(
        config,
        orientation,
        random_streams.initialization(),
    )
    components = assemble_baseline_components(
        config,
        initializer=FixedBaselineInitializer(initialization.state),
        leader_ids=initialization.leader_ids,
    )
    result = run_simulation(config.simulation, components)
    metrics = baseline_round_metrics(
        result,
        seed=config.simulation.seed,
        orientation=orientation,
        leader_ids=initialization.leader_ids,
        extremism_threshold=extremism_threshold,
    )
    return BaselineRunResult(
        initialization=initialization,
        simulation_result=result,
        round_metrics=metrics,
    )


def run_baseline_experiment(
    experiment: BaselineExperimentConfig,
) -> pd.DataFrame:
    """Run every matched seed-orientation combination in the configuration."""
    frames = []
    for seed in experiment.seeds:
        run_config = replace(
            experiment.baseline,
            simulation=replace(experiment.baseline.simulation, seed=seed),
        )
        for orientation in experiment.orientations:
            run = run_baseline_condition(
                run_config,
                orientation,
                extremism_threshold=experiment.extremism_threshold,
            )
            frames.append(run.round_metrics)
    return pd.concat(frames, ignore_index=True)


def final_condition_metrics(round_metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two mirrored balanced assignments within each seed."""
    final_round = int(round_metrics["round"].max())
    final = round_metrics.loc[round_metrics["round"] == final_round].copy()
    metric_columns = [
        column
        for column in final.columns
        if column not in {"seed", "orientation", "round"}
    ]
    directional = final.loc[
        final["orientation"].isin(["positive", "negative"])
    ].copy()
    directional["condition"] = directional["orientation"]
    balanced = (
        final.loc[
            final["orientation"].isin(
                ["balanced_positive", "balanced_negative"]
            )
        ]
        .groupby("seed", as_index=False)[metric_columns]
        .mean()
    )
    balanced["orientation"] = "balanced_pair_mean"
    balanced["condition"] = "balanced"
    columns = ["seed", "condition", "orientation", "round", *metric_columns]
    balanced["round"] = final_round
    return pd.concat(
        [directional[columns], balanced[columns]],
        ignore_index=True,
    ).sort_values(["seed", "condition"], ignore_index=True)


def summarize_final_conditions(final_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return means and sample standard deviations across matched seeds."""
    metric_columns = [
        "mean_signed_belief",
        "mean_absolute_belief",
        "extremist_ratio",
        "ordinary_mean_signed_belief",
        "leader_mean_signed_belief",
        "cumulative_content_balance",
        "edge_count",
        "mean_following_degree",
        "homophily_ratio",
    ]
    summary = final_metrics.groupby("condition")[metric_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    summary.insert(0, "run_count", final_metrics.groupby("condition").size())
    return summary.reset_index()


__all__ = [
    "BaselineRunResult",
    "final_condition_metrics",
    "run_baseline_condition",
    "run_baseline_experiment",
    "summarize_final_conditions",
]
