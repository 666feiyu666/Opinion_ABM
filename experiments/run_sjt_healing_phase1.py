from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_SEED, make_params
from main import run_simulation
from metrics import summarize_metric_distribution
from utils import ensure_directory, project_path
from visualization import (
    plot_baseline_validation,
    plot_healing_curves,
    plot_pre_post_comparison,
)

SWEEP_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
CONTRAST_THRESHOLDS = (0.2, 0.8)


def phase1_baseline_overrides(rounds: int, tolerance_threshold: float = 0.1) -> dict:
    return {
        "T_rounds": rounds,
        "tolerance_threshold": tolerance_threshold,
        # Keep single-round updates in a smooth regime.
        "omega_pC_in": 0.05,
        "omega_pT_in": 0.01,
        "omega_nC_in": -0.04,
        "omega_nT_in": 0.01,
        "omega_pC_out": 0.08,
        "omega_pT_out": 0.02,
        "omega_nC_out": -0.005,
        "omega_nT_out": 0.06,
        "omega_pC_in_L": 0.03,
        "omega_pT_in_L": 0.008,
        "omega_nC_in_L": -0.025,
        "omega_nT_in_L": 0.008,
        "omega_pC_out_L": 0.05,
        "omega_pT_out_L": 0.015,
        "omega_nC_out_L": -0.003,
        "omega_nT_out_L": 0.035,
        # Keep attention capacity in a human-bounded range.
        "max_read_capacity": 8,
        # Maintain moderate algorithmic toxicity as the polarization engine.
        "beta2_diff": 0.70,
    }


def run_single_condition(params: dict, seed: int, track_opinions: bool = False) -> dict:
    return run_simulation(
        params=params,
        seed=seed,
        track_opinions=track_opinions,
    )


def save_figure(fig, output_path: Path):
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def aggregate_sweep(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    metric_columns = [
        "final_opinion_variance",
        "final_extremist_ratio",
        "final_homophily_ratio",
        "final_sign_modularity",
        "final_cross_cutting_ratio",
        "final_edge_count",
    ]

    rename_map = {
        "final_opinion_variance": "opinion_variance",
        "final_extremist_ratio": "extremist_ratio",
        "final_homophily_ratio": "homophily_ratio",
        "final_sign_modularity": "sign_modularity",
        "final_cross_cutting_ratio": "cross_cutting_ratio",
        "final_edge_count": "edge_count",
    }

    for threshold, threshold_df in raw_df.groupby("tolerance_threshold"):
        row = {"tolerance_threshold": float(threshold)}
        for column in metric_columns:
            metric_summary = summarize_metric_distribution(threshold_df[column].tolist())
            metric_name = rename_map[column]
            row[f"{metric_name}_mean"] = metric_summary["mean"]
            row[f"{metric_name}_std"] = metric_summary["std"]
            row[f"{metric_name}_sem"] = metric_summary["sem"]
            row[f"{metric_name}_ci_low"] = metric_summary["ci_low"]
            row[f"{metric_name}_ci_high"] = metric_summary["ci_high"]
            row[f"{metric_name}_n"] = metric_summary["n"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("tolerance_threshold").reset_index(drop=True)
    baseline_variance = float(summary_df.iloc[0]["opinion_variance_mean"])
    baseline_extremism = float(summary_df.iloc[0]["extremist_ratio_mean"])
    summary_df["opinion_variance_drop_vs_baseline"] = baseline_variance - summary_df["opinion_variance_mean"]
    summary_df["extremist_ratio_drop_vs_baseline"] = baseline_extremism - summary_df["extremist_ratio_mean"]
    return summary_df


def run_phase1_experiment(
    rounds: int = 120,
    seeds: int = 10,
    output_subdir: str = "phase1_sjt_healing",
):
    output_dir = ensure_directory(project_path("outputs", output_subdir))
    seed_values = [DEFAULT_SEED + offset for offset in range(seeds)]

    baseline_params = make_params(phase1_baseline_overrides(rounds=rounds, tolerance_threshold=0.1))
    baseline_result = run_single_condition(
        params=baseline_params,
        seed=DEFAULT_SEED,
        track_opinions=True,
    )

    baseline_result["history_df"].to_csv(output_dir / "baseline_history.csv", index=False)
    baseline_result["opinion_trajectory_df"].to_csv(output_dir / "baseline_opinion_trajectories.csv", index=False)
    baseline_result["agents"].to_csv(output_dir / "baseline_agents_final.csv", index=False)

    fig, _ = plot_baseline_validation(
        baseline_result["history_df"],
        baseline_result["opinion_trajectory_df"],
        baseline_result["G_updated"],
        baseline_result["pos"],
    )
    save_figure(fig, output_dir / "baseline_validation.png")

    sweep_rows = []
    for threshold in SWEEP_THRESHOLDS:
        params = make_params(phase1_baseline_overrides(rounds=rounds, tolerance_threshold=threshold))
        print(f"Running threshold={threshold:.1f} across {len(seed_values)} seeds...")
        for seed in seed_values:
            result = run_single_condition(params=params, seed=seed, track_opinions=False)
            sweep_rows.append(
                {
                    "seed": int(seed),
                    "tolerance_threshold": float(threshold),
                    **result["final_state"],
                }
            )

    sweep_raw_df = pd.DataFrame(sweep_rows).sort_values(["tolerance_threshold", "seed"]).reset_index(drop=True)
    sweep_summary_df = aggregate_sweep(sweep_raw_df)
    sweep_raw_df.to_csv(output_dir / "sweep_raw.csv", index=False)
    sweep_summary_df.to_csv(output_dir / "sweep_summary.csv", index=False)

    fig, _ = plot_healing_curves(sweep_summary_df)
    save_figure(fig, output_dir / "healing_curves.png")

    pathology_threshold, healing_threshold = CONTRAST_THRESHOLDS
    pathology_result = run_single_condition(
        params=make_params(phase1_baseline_overrides(rounds=rounds, tolerance_threshold=pathology_threshold)),
        seed=DEFAULT_SEED,
        track_opinions=False,
    )
    healing_result = run_single_condition(
        params=make_params(phase1_baseline_overrides(rounds=rounds, tolerance_threshold=healing_threshold)),
        seed=DEFAULT_SEED,
        track_opinions=False,
    )

    pathology_result["agents"].to_csv(output_dir / "contrast_pathology_agents_final.csv", index=False)
    healing_result["agents"].to_csv(output_dir / "contrast_healing_agents_final.csv", index=False)

    contrast_summary_df = pd.DataFrame(
        [
            {"condition": f"pathology_{pathology_threshold:.1f}", **pathology_result["final_state"]},
            {"condition": f"healing_{healing_threshold:.1f}", **healing_result["final_state"]},
        ]
    )
    contrast_summary_df.to_csv(output_dir / "contrast_summary.csv", index=False)

    fig, _ = plot_pre_post_comparison(
        pathology_result=pathology_result,
        healing_result=healing_result,
        pathology_threshold=pathology_threshold,
        healing_threshold=healing_threshold,
    )
    save_figure(fig, output_dir / "pre_post_comparison.png")

    experiment_manifest = pd.DataFrame(
        [
            {
                "rounds": rounds,
                "seeds": seeds,
                "default_seed": DEFAULT_SEED,
                "sweep_thresholds": ",".join(str(value) for value in SWEEP_THRESHOLDS),
                "contrast_thresholds": ",".join(str(value) for value in CONTRAST_THRESHOLDS),
            }
        ]
    )
    experiment_manifest.to_csv(output_dir / "experiment_manifest.csv", index=False)

    print(f"Saved Phase 1 SJT healing outputs to: {output_dir}")
    return {
        "output_dir": output_dir,
        "baseline_result": baseline_result,
        "sweep_raw_df": sweep_raw_df,
        "sweep_summary_df": sweep_summary_df,
        "pathology_result": pathology_result,
        "healing_result": healing_result,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 1 SJT healing experiment.")
    parser.add_argument("--rounds", type=int, default=120, help="Simulation rounds per run.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds for the sweep.")
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="phase1_sjt_healing",
        help="Subdirectory under outputs/ for saved figures and CSV files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_phase1_experiment(
        rounds=args.rounds,
        seeds=args.seeds,
        output_subdir=args.output_subdir,
    )
