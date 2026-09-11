"""Matched execution of the exploratory null scenario."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from opinion_model.scenarios.null.assembly import assemble_null_components
from opinion_model.scenarios.null.config import NullConfig, NullExperimentConfig
from opinion_model.scenarios.null.initialization import (
    FixedNullInitializer,
    NullInitialization,
    initialize_null,
)
from opinion_model.scenarios.null.observation import null_round_metrics
from opinion_model.shared import RandomStreams, SimulationResult, run_simulation


@dataclass(frozen=True)
class NullRunResult:
    """One resolved initialization, simulation result, and diagnostic table."""

    initialization: NullInitialization
    simulation_result: SimulationResult
    round_metrics: pd.DataFrame


def run_null_condition(
    config: NullConfig,
    *,
    extremism_threshold: float,
) -> NullRunResult:
    """Run one null condition from its matched initialized world."""
    random_streams = RandomStreams(config.simulation.seed)
    initialization = initialize_null(
        config,
        random_streams.initialization(),
    )
    components = assemble_null_components(
        config,
        initializer=FixedNullInitializer(initialization.state),
    )
    result = run_simulation(config.simulation, components)
    metrics = null_round_metrics(
        result,
        seed=config.simulation.seed,
        extremism_threshold=extremism_threshold,
    )
    return NullRunResult(
        initialization=initialization,
        simulation_result=result,
        round_metrics=metrics,
    )


def run_null_experiment(experiment: NullExperimentConfig) -> pd.DataFrame:
    """Run one null condition for every configured seed."""
    frames = []
    for seed in experiment.seeds:
        run_config = replace(
            experiment.null,
            simulation=replace(experiment.null.simulation, seed=seed),
        )
        run = run_null_condition(
            run_config,
            extremism_threshold=experiment.extremism_threshold,
        )
        frames.append(run.round_metrics)
    return pd.concat(frames, ignore_index=True)


def final_condition_metrics(round_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the final round of each matched null run."""
    final_round = int(round_metrics["round"].max())
    return (
        round_metrics.loc[round_metrics["round"] == final_round]
        .copy()
        .sort_values(["seed", "condition"], ignore_index=True)
    )


def summarize_final_conditions(final_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return means and sample standard deviations across matched seeds."""
    metric_columns = [
        "mean_signed_belief",
        "mean_absolute_belief",
        "extremist_ratio",
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
    "NullRunResult",
    "final_condition_metrics",
    "run_null_condition",
    "run_null_experiment",
    "summarize_final_conditions",
]
