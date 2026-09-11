"""Matched execution of the exploratory platform-only scenario."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from opinion_model.scenarios.platform.assembly import assemble_platform_components
from opinion_model.scenarios.platform.config import (
    PlatformConfig,
    PlatformExperimentConfig,
)
from opinion_model.scenarios.platform.initialization import (
    FixedPlatformInitializer,
    PlatformInitialization,
    initialize_platform,
)
from opinion_model.scenarios.platform.observation import platform_round_metrics
from opinion_model.shared import RandomStreams, SimulationResult, run_simulation


@dataclass(frozen=True)
class PlatformRunResult:
    """One resolved initialization, simulation result, and diagnostic table."""

    initialization: PlatformInitialization
    simulation_result: SimulationResult
    round_metrics: pd.DataFrame


def run_platform_condition(
    config: PlatformConfig,
    *,
    extremism_threshold: float,
    agent_order: tuple[int, ...] | None = None,
) -> PlatformRunResult:
    """Run one platform condition from its matched initialized world."""
    random_streams = RandomStreams(config.simulation.seed)
    initialization = initialize_platform(
        config,
        random_streams.initialization(),
    )
    components = assemble_platform_components(
        config,
        initializer=FixedPlatformInitializer(initialization.state),
    )
    result = run_simulation(
        config.simulation,
        components,
        agent_order=agent_order,
    )
    metrics = platform_round_metrics(
        result,
        config,
        extremism_threshold=extremism_threshold,
    )
    return PlatformRunResult(
        initialization=initialization,
        simulation_result=result,
        round_metrics=metrics,
    )


def run_platform_experiment(
    experiment: PlatformExperimentConfig,
) -> pd.DataFrame:
    """Run the platform condition once for every configured seed."""
    frames = []
    for seed in experiment.seeds:
        run_config = replace(
            experiment.platform_case,
            simulation=replace(
                experiment.platform_case.simulation,
                seed=seed,
            ),
        )
        run = run_platform_condition(
            run_config,
            extremism_threshold=experiment.extremism_threshold,
        )
        frames.append(run.round_metrics)
    return pd.concat(frames, ignore_index=True)


def final_condition_metrics(round_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the final round of each matched platform run."""
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
        "exposure_count",
        "tied_exposure_count",
        "out_of_network_exposure_count",
        "capacity_binding_rate",
        "cumulative_content_balance",
        "formation_opportunity_count",
        "dissolution_opportunity_count",
        "accepted_addition_count",
        "accepted_removal_count",
        "edge_count",
        "mean_following_degree",
        "isolated_agent_count",
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
    "PlatformRunResult",
    "final_condition_metrics",
    "run_platform_condition",
    "run_platform_experiment",
    "summarize_final_conditions",
]
