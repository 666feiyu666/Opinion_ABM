"""Matched execution of the exploratory opinion-leader-only scenario."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from opinion_model.scenarios.opleader.assembly import assemble_opleader_components
from opinion_model.scenarios.opleader.config import (
    OpleaderConfig,
    OpleaderExperimentConfig,
    OpleaderOrientation,
)
from opinion_model.scenarios.opleader.initialization import (
    FixedOpleaderInitializer,
    OpleaderInitialization,
    initialize_opleader,
)
from opinion_model.scenarios.opleader.observation import opleader_round_metrics
from opinion_model.shared import RandomStreams, SimulationResult, run_simulation


@dataclass(frozen=True)
class OpleaderRunResult:
    """One resolved initialization, simulation result, and metric table."""

    initialization: OpleaderInitialization
    simulation_result: SimulationResult
    round_metrics: pd.DataFrame


def run_opleader_condition(
    config: OpleaderConfig,
    orientation: OpleaderOrientation,
    *,
    extremism_threshold: float,
) -> OpleaderRunResult:
    """Run one orientation from its seed-matched initialized world."""
    random_streams = RandomStreams(config.simulation.seed)
    initialization = initialize_opleader(
        config,
        orientation,
        random_streams.initialization(),
    )
    components = assemble_opleader_components(
        config,
        initializer=FixedOpleaderInitializer(initialization.state),
        leader_ids=initialization.leader_ids,
    )
    result = run_simulation(config.simulation, components)
    metrics = opleader_round_metrics(
        result,
        seed=config.simulation.seed,
        orientation=orientation,
        leader_ids=initialization.leader_ids,
        extremism_threshold=extremism_threshold,
    )
    return OpleaderRunResult(
        initialization=initialization,
        simulation_result=result,
        round_metrics=metrics,
    )


def run_opleader_experiment(
    experiment: OpleaderExperimentConfig,
) -> pd.DataFrame:
    """Run every matched seed-orientation combination in the configuration."""
    frames = []
    for seed in experiment.seeds:
        run_config = replace(
            experiment.opleader,
            simulation=replace(experiment.opleader.simulation, seed=seed),
        )
        for orientation in experiment.orientations:
            run = run_opleader_condition(
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
    balanced["round"] = final_round
    columns = ["seed", "condition", "orientation", "round", *metric_columns]
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
        "leader_origination_rate",
        "ordinary_origination_rate",
        "leader_exposure_share",
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
    "OpleaderRunResult",
    "final_condition_metrics",
    "run_opleader_condition",
    "run_opleader_experiment",
    "summarize_final_conditions",
]
