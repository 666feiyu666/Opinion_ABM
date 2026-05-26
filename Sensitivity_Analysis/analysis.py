from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import summarize_metric_distribution
from utils import ensure_directory

EFFECT_METRICS = [
    "final_mean_opinion",
    "final_mean_abs_opinion",
    "extremist_ratio",
    "homophily_ratio",
    "sign_modularity",
    "content_balance",
]

SUMMARY_GROUP_COLUMNS = [
    "profile_name",
    "study_name",
    "N",
    "topology",
    "leader_share",
    "leader_mode",
    "T_rounds",
    "perturbation_id",
    "varied_parameter",
    "parameter_label",
    "parameter_level",
    "parameter_value",
]


def compute_matched_effects(raw_df: pd.DataFrame) -> pd.DataFrame:
    controls = raw_df[raw_df["condition_role"] == "control"].copy()
    leaders = raw_df[raw_df["condition_role"] == "leader"].copy()
    control_columns = ["condition_id", *EFFECT_METRICS]
    controls = controls[control_columns].rename(
        columns={
            "condition_id": "matched_control_id",
            **{metric: f"control_{metric}" for metric in EFFECT_METRICS},
        }
    )
    matched = leaders.merge(controls, on="matched_control_id", how="left", validate="many_to_one")
    for metric in EFFECT_METRICS:
        matched[f"delta_{metric}"] = matched[metric] - matched[f"control_{metric}"]
    return matched


def _summarize_groups(frame: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for keys, group_df in frame.groupby(SUMMARY_GROUP_COLUMNS, dropna=False):
        row = dict(zip(SUMMARY_GROUP_COLUMNS, keys))
        for metric in metric_columns:
            stats = summarize_metric_distribution(group_df[metric].to_numpy())
            for statistic, value in stats.items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(SUMMARY_GROUP_COLUMNS).reset_index(drop=True)


def aggregate_effects(matched_effects_df: pd.DataFrame) -> pd.DataFrame:
    metrics = EFFECT_METRICS + [f"delta_{metric}" for metric in EFFECT_METRICS]
    return _summarize_groups(matched_effects_df, metrics)


def aggregate_controls(raw_df: pd.DataFrame) -> pd.DataFrame:
    controls = raw_df[raw_df["condition_role"] == "control"].copy()
    if controls.empty:
        return pd.DataFrame()
    group_columns = ["profile_name", "study_name", "N", "topology", "T_rounds"]
    rows = []
    for keys, group_df in controls.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        for metric in EFFECT_METRICS:
            stats = summarize_metric_distribution(group_df[metric].to_numpy())
            for statistic, value in stats.items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def _status(condition: bool) -> str:
    return "preserved" if condition else "not_preserved"


def build_robustness_checks(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "study_name" not in summary_df.columns:
        return pd.DataFrame()
    check_rows = []
    core = summary_df[summary_df["study_name"] == "core_robustness"]
    group_columns = ["profile_name", "study_name", "perturbation_id", "topology", "leader_share"]
    for keys, group_df in core.groupby(group_columns, dropna=False):
        by_mode = group_df.set_index("leader_mode")
        if not {"balanced", "positive", "negative"}.issubset(by_mode.index):
            continue
        positive = by_mode.loc["positive"]
        negative = by_mode.loc["negative"]
        balanced = by_mode.loc["balanced"]
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "check_scope": "benchmark",
                "directional_order": _status(
                    positive["delta_final_mean_opinion_mean"] > 0
                    and negative["delta_final_mean_opinion_mean"] < 0
                ),
                "directional_ci_support": _status(
                    positive["delta_final_mean_opinion_ci_low"] > 0
                    and negative["delta_final_mean_opinion_ci_high"] < 0
                ),
                "one_sided_extremity": _status(
                    positive["delta_extremist_ratio_mean"] > 0
                    and negative["delta_extremist_ratio_mean"] > 0
                ),
                "extremity_ci_support": _status(
                    positive["delta_extremist_ratio_ci_low"] > 0
                    and negative["delta_extremist_ratio_ci_low"] > 0
                ),
                "one_sided_homophily": _status(
                    positive["delta_homophily_ratio_mean"] > 0
                    and negative["delta_homophily_ratio_mean"] > 0
                ),
                "homophily_ci_support": _status(
                    positive["delta_homophily_ratio_ci_low"] > 0
                    and negative["delta_homophily_ratio_ci_low"] > 0
                ),
                "balanced_direction_weaker": _status(
                    abs(balanced["delta_final_mean_opinion_mean"])
                    < abs(positive["delta_final_mean_opinion_mean"])
                    and abs(balanced["delta_final_mean_opinion_mean"])
                    < abs(negative["delta_final_mean_opinion_mean"])
                ),
                "directional_endpoint_increase": "not_applicable",
                "extremity_endpoint_increase": "not_applicable",
                "homophily_endpoint_increase": "not_applicable",
            }
        )
        check_rows.append(row)

    gradient = summary_df[summary_df["study_name"] == "share_gradient"]
    group_columns = ["profile_name", "study_name", "perturbation_id", "topology", "leader_mode"]
    for keys, group_df in gradient.groupby(group_columns, dropna=False):
        if keys[-1] not in {"positive", "negative"} or len(group_df) < 3:
            continue
        ordered = group_df.sort_values("leader_share")
        directional_effect = ordered["delta_final_mean_opinion_mean"].to_numpy()
        if keys[-1] == "negative":
            directional_effect = -directional_effect
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "check_scope": "share_gradient",
                "directional_order": _status(np.all(np.diff(directional_effect) >= 0)),
                "directional_ci_support": "not_applicable",
                "one_sided_extremity": _status(
                    np.all(np.diff(ordered["delta_extremist_ratio_mean"].to_numpy()) >= 0)
                ),
                "extremity_ci_support": "not_applicable",
                "one_sided_homophily": _status(
                    np.all(np.diff(ordered["delta_homophily_ratio_mean"].to_numpy()) >= 0)
                ),
                "homophily_ci_support": "not_applicable",
                "balanced_direction_weaker": "not_applicable",
                "directional_endpoint_increase": _status(directional_effect[-1] >= directional_effect[0]),
                "extremity_endpoint_increase": _status(
                    ordered["delta_extremist_ratio_mean"].iloc[-1]
                    >= ordered["delta_extremist_ratio_mean"].iloc[0]
                ),
                "homophily_endpoint_increase": _status(
                    ordered["delta_homophily_ratio_mean"].iloc[-1]
                    >= ordered["delta_homophily_ratio_mean"].iloc[0]
                ),
            }
        )
        check_rows.append(row)
    return pd.DataFrame(check_rows)


def export_outputs(
    output_root: str | Path,
    grid_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    matched_effects_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    control_summary_df: pd.DataFrame,
    checks_df: pd.DataFrame,
    *,
    profile_name: str,
    study_name: str,
    notes: str | None = None,
) -> dict[str, Path]:
    output_dir = ensure_directory(output_root)
    paths = {
        "grid": output_dir / "experiment_grid.csv",
        "raw": output_dir / "raw_results.csv",
        "matched_effects": output_dir / "matched_effects.csv",
        "summary": output_dir / "summary_effects.csv",
        "control_summary": output_dir / "control_summary.csv",
        "checks": output_dir / "robustness_checks.csv",
        "manifest": output_dir / "manifest.json",
    }
    grid_df.to_csv(paths["grid"], index=False)
    raw_df.to_csv(paths["raw"], index=False)
    matched_effects_df.to_csv(paths["matched_effects"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    control_summary_df.to_csv(paths["control_summary"], index=False)
    checks_df.to_csv(paths["checks"], index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_name": profile_name,
        "study_name": study_name,
        "planned_run_count": int(len(grid_df)),
        "completed_run_count": int(len(raw_df)),
        "leader_run_count": int((grid_df["condition_role"] == "leader").sum()),
        "control_run_count": int((grid_df["condition_role"] == "control").sum()),
        "topologies": sorted(grid_df["topology"].unique().tolist()),
        "leader_modes": sorted(grid_df["leader_mode"].unique().tolist()),
        "leader_shares": sorted(grid_df["leader_share"].unique().tolist()),
        "rounds": sorted(grid_df["T_rounds"].unique().tolist()),
        "seeds": sorted(grid_df["seed"].unique().tolist()),
        "perturbations": sorted(grid_df["perturbation_id"].unique().tolist()),
        "comparison_basis": "Each leader run is differenced against its topology- and seed-matched no-leader control.",
        "notes": notes,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    from Sensitivity_Analysis.plots import save_sensitivity_figures

    paths.update(save_sensitivity_figures(summary_df, output_dir))
    return paths
