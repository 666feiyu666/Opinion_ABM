from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opinion_abm_mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_SEED, make_notebook_baseline_params
from main import run_simulation
from metrics import summarize_metric_distribution
from utils import ensure_directory, project_path
from visualization import plot_final_opinion_heatmap, plot_tolerance_sweep_curves

SWEEP_THRESHOLDS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
MODERATE_BAND = 0.3


def phase1_overrides(rounds: int, tolerance_threshold: float) -> dict:
    return {
        "T_rounds": rounds,
        "tolerance_threshold": tolerance_threshold,
    }


def build_seed_values(seeds: int, seed_start: int) -> list[int]:
    return [int(seed_start + offset) for offset in range(seeds)]


def run_single_condition(threshold: float, seed: int, rounds: int) -> tuple[dict, list[dict]]:
    params = make_notebook_baseline_params(
        phase1_overrides(rounds=rounds, tolerance_threshold=threshold)
    )
    result = run_simulation(params=params, seed=seed, track_opinions=False)

    summary_row = {
        "seed": int(seed),
        "tolerance_threshold": float(threshold),
        **result["final_state"],
    }

    opinion_rows = [
        {
            "seed": int(seed),
            "tolerance_threshold": float(threshold),
            "node": int(node),
            "opinion": float(opinion),
            "is_leader": int(is_leader),
        }
        for node, opinion, is_leader in zip(
            result["agents"]["node"],
            result["agents"]["o_t"],
            result["agents"]["L"],
        )
    ]
    return summary_row, opinion_rows


def _run_single_condition_from_task(task: tuple[float, int, int]) -> tuple[dict, list[dict]]:
    threshold, seed, rounds = task
    return run_single_condition(threshold=threshold, seed=seed, rounds=rounds)


def save_figure(fig, output_path: Path):
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def aggregate_sweep(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    metric_columns = [
        "final_mean_opinion",
        "final_std_opinion",
        "final_mean_abs_opinion",
        "final_opinion_variance",
        "final_extremist_ratio",
        "final_moderate_ratio",
        "final_homophily_ratio",
        "final_cross_cutting_ratio",
        "final_sign_modularity",
        "final_edge_count",
    ]

    rename_map = {
        "final_mean_opinion": "mean_opinion",
        "final_std_opinion": "std_opinion",
        "final_mean_abs_opinion": "mean_abs_opinion",
        "final_opinion_variance": "opinion_variance",
        "final_extremist_ratio": "extremist_ratio",
        "final_moderate_ratio": "moderate_ratio",
        "final_homophily_ratio": "homophily_ratio",
        "final_cross_cutting_ratio": "cross_cutting_ratio",
        "final_sign_modularity": "sign_modularity",
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

    return pd.DataFrame(summary_rows).sort_values("tolerance_threshold").reset_index(drop=True)


def run_phase1_experiment(
    rounds: int = 50,
    seeds: int = 20,
    output_subdir: str = "phase1_tolerance_sweep",
    workers: int = 1,
    seed_start: int = DEFAULT_SEED,
):
    output_dir = ensure_directory(project_path("outputs", output_subdir))
    seed_values = build_seed_values(seeds=seeds, seed_start=seed_start)
    tasks = [
        (float(threshold), int(seed), int(rounds))
        for threshold in SWEEP_THRESHOLDS
        for seed in seed_values
    ]

    sweep_rows = []
    opinion_rows = []

    if workers > 1:
        max_workers = min(workers, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for summary_row, final_opinions in executor.map(_run_single_condition_from_task, tasks):
                sweep_rows.append(summary_row)
                opinion_rows.extend(final_opinions)
    else:
        for threshold in SWEEP_THRESHOLDS:
            print(f"Running theta_T={threshold:.1f} across {len(seed_values)} seeds...")
            for seed in seed_values:
                summary_row, final_opinions = run_single_condition(
                    threshold=threshold,
                    seed=seed,
                    rounds=rounds,
                )
                sweep_rows.append(summary_row)
                opinion_rows.extend(final_opinions)

    sweep_raw_df = pd.DataFrame(sweep_rows).sort_values(["tolerance_threshold", "seed"]).reset_index(drop=True)
    sweep_summary_df = aggregate_sweep(sweep_raw_df)
    opinion_samples_df = pd.DataFrame(opinion_rows).sort_values(
        ["tolerance_threshold", "seed", "node"]
    ).reset_index(drop=True)

    sweep_raw_df.to_csv(output_dir / "sweep_raw.csv", index=False)
    sweep_summary_df.to_csv(output_dir / "sweep_summary.csv", index=False)
    opinion_samples_df.to_csv(output_dir / "final_opinion_samples.csv", index=False)

    fig, _ = plot_tolerance_sweep_curves(sweep_summary_df)
    save_figure(fig, output_dir / "tolerance_sweep_curves.png")

    fig, _ = plot_final_opinion_heatmap(opinion_samples_df)
    save_figure(fig, output_dir / "final_opinion_heatmap.png")

    experiment_manifest = pd.DataFrame(
        [
            {
                "phase": "phase1_robust_baseline",
                "rounds": rounds,
                "seeds": seeds,
                "default_seed": DEFAULT_SEED,
                "seed_start": seed_start,
                "moderate_band": MODERATE_BAND,
                "sweep_thresholds": ",".join(str(value) for value in SWEEP_THRESHOLDS),
                "workers": workers,
            }
        ]
    )
    experiment_manifest.to_csv(output_dir / "experiment_manifest.csv", index=False)

    compact_summary = sweep_summary_df[
        [
            "tolerance_threshold",
            "extremist_ratio_mean",
            "opinion_variance_mean",
            "moderate_ratio_mean",
            "mean_abs_opinion_mean",
        ]
    ].copy()
    print(compact_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved Phase 1 tolerance sweep outputs to: {output_dir}")

    return {
        "output_dir": output_dir,
        "sweep_raw_df": sweep_raw_df,
        "sweep_summary_df": sweep_summary_df,
        "opinion_samples_df": opinion_samples_df,
    }


def merge_phase1_batches(input_subdirs: list[str], output_subdir: str):
    output_dir = ensure_directory(project_path("outputs", output_subdir))

    raw_frames = []
    opinion_frames = []
    batch_manifests = []

    for subdir in input_subdirs:
        batch_dir = project_path("outputs", subdir)
        raw_frames.append(pd.read_csv(batch_dir / "sweep_raw.csv"))
        opinion_frames.append(pd.read_csv(batch_dir / "final_opinion_samples.csv"))

        manifest_path = batch_dir / "experiment_manifest.csv"
        if manifest_path.exists():
            manifest_df = pd.read_csv(manifest_path)
            if not manifest_df.empty:
                batch_manifests.append(manifest_df.iloc[0].to_dict())

    sweep_raw_df = pd.concat(raw_frames, ignore_index=True)
    sweep_raw_df = sweep_raw_df.sort_values(["tolerance_threshold", "seed"]).reset_index(drop=True)

    opinion_samples_df = pd.concat(opinion_frames, ignore_index=True)
    opinion_samples_df = opinion_samples_df.sort_values(
        ["tolerance_threshold", "seed", "node"]
    ).reset_index(drop=True)

    sweep_summary_df = aggregate_sweep(sweep_raw_df)

    sweep_raw_df.to_csv(output_dir / "sweep_raw.csv", index=False)
    sweep_summary_df.to_csv(output_dir / "sweep_summary.csv", index=False)
    opinion_samples_df.to_csv(output_dir / "final_opinion_samples.csv", index=False)

    fig, _ = plot_tolerance_sweep_curves(sweep_summary_df)
    save_figure(fig, output_dir / "tolerance_sweep_curves.png")

    fig, _ = plot_final_opinion_heatmap(opinion_samples_df)
    save_figure(fig, output_dir / "final_opinion_heatmap.png")

    merged_manifest = pd.DataFrame(
        [
            {
                "phase": "phase1_robust_baseline",
                "input_subdirs": ",".join(input_subdirs),
                "batch_count": len(input_subdirs),
                "seed_count": int(sweep_raw_df["seed"].nunique()),
                "sweep_thresholds": ",".join(str(value) for value in SWEEP_THRESHOLDS),
                "moderate_band": MODERATE_BAND,
                "batch_seed_starts": ",".join(
                    str(int(manifest["seed_start"]))
                    for manifest in batch_manifests
                    if "seed_start" in manifest and pd.notna(manifest["seed_start"])
                ),
                "rounds": int(batch_manifests[0]["rounds"]) if batch_manifests else None,
            }
        ]
    )
    merged_manifest.to_csv(output_dir / "experiment_manifest.csv", index=False)

    compact_summary = sweep_summary_df[
        [
            "tolerance_threshold",
            "extremist_ratio_mean",
            "opinion_variance_mean",
            "moderate_ratio_mean",
            "mean_abs_opinion_mean",
        ]
    ].copy()
    print(compact_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved merged Phase 1 tolerance sweep outputs to: {output_dir}")

    return {
        "output_dir": output_dir,
        "sweep_raw_df": sweep_raw_df,
        "sweep_summary_df": sweep_summary_df,
        "opinion_samples_df": opinion_samples_df,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 1 tolerance sweep experiment.")
    parser.add_argument("--rounds", type=int, default=50, help="Simulation rounds per run.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of Monte Carlo repetitions per threshold.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED,
        help="Starting random seed for the batch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes. Use 1 for deterministic serial execution.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="phase1_tolerance_sweep",
        help="Subdirectory under outputs/ for saved figures and CSV files.",
    )
    parser.add_argument(
        "--merge-input-subdirs",
        nargs="+",
        help="Merge existing batch output subdirectories instead of running new simulations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.merge_input_subdirs:
        merge_phase1_batches(
            input_subdirs=args.merge_input_subdirs,
            output_subdir=args.output_subdir,
        )
    else:
        run_phase1_experiment(
            rounds=args.rounds,
            seeds=args.seeds,
            output_subdir=args.output_subdir,
            workers=args.workers,
            seed_start=args.seed_start,
        )
