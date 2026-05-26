from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.base import build_simulation_params, extract_condition_result
from main import run_simulation
from Sensitivity_Analysis.analysis import (
    aggregate_controls,
    aggregate_effects,
    build_robustness_checks,
    compute_matched_effects,
    export_outputs,
)
from Sensitivity_Analysis.design import build_sensitivity_grid, parameter_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OLIM leader-mechanism robustness sensitivity studies.")
    parser.add_argument(
        "--profile",
        default="main",
        choices=["smoke", "trial", "main"],
        help="smoke verifies execution; trial and main retain the full study structure.",
    )
    parser.add_argument(
        "--study",
        default="core_robustness",
        choices=["core_robustness", "share_gradient", "all"],
        help="Robustness study to run.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional truncation for development checks.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel simulation worker processes.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write partial raw results after this many newly completed runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from raw_results.partial.csv or raw_results.csv in the output directory.",
    )
    return parser.parse_args()


def build_grid(profile_name: str, study_name: str) -> pd.DataFrame:
    studies = ["core_robustness", "share_gradient"] if study_name == "all" else [study_name]
    return pd.concat(
        [build_sensitivity_grid(profile_name, selected_study) for selected_study in studies],
        ignore_index=True,
    )


def run_condition(condition: dict) -> dict:
    extra_overrides = parameter_overrides(condition["perturbation_id"], condition["study_name"])
    params = build_simulation_params(condition, extra_overrides=extra_overrides)
    results = run_simulation(
        params=params,
        seed=int(condition["seed"]),
        rounds=int(condition["T_rounds"]),
        compute_layout=False,
        retain_round_details=False,
    )
    return extract_condition_result(condition, results)


def default_output_dir(profile_name: str, study_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "sensitivity_analysis" / f"{study_name}_{profile_name}"


def _order_raw_results(raw_df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    condition_order = pd.Categorical(raw_df["condition_id"], categories=grid_df["condition_id"], ordered=True)
    return raw_df.assign(_condition_order=condition_order).sort_values("_condition_order").drop(columns="_condition_order")


def _load_completed_results(output_dir: Path, resume: bool) -> pd.DataFrame:
    if not resume:
        return pd.DataFrame()
    for path in [output_dir / "raw_results.partial.csv", output_dir / "raw_results.csv"]:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def main():
    args = parse_args()
    grid_df = build_grid(args.profile, args.study)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.profile, args.study)
    output_dir.mkdir(parents=True, exist_ok=True)
    notes = None
    if args.max_runs is not None:
        grid_df = grid_df.head(int(args.max_runs)).copy()
        notes = f"Run truncated with --max-runs={args.max_runs}; checks may be incomplete."

    completed_df = _load_completed_results(output_dir, args.resume)
    completed_ids = set(completed_df["condition_id"]) if not completed_df.empty else set()
    pending_df = grid_df[~grid_df["condition_id"].isin(completed_ids)].copy()
    raw_rows = completed_df.to_dict("records") if not completed_df.empty else []
    total_runs = len(grid_df)
    print(
        f"Prepared {total_runs} runs; {len(completed_ids)} already complete; "
        f"{len(pending_df)} pending; workers={max(1, int(args.workers))}."
    )
    checkpoint_path = output_dir / "raw_results.partial.csv"
    checkpoint_every = max(1, int(args.checkpoint_every))
    newly_completed = 0

    def record_result(result: dict):
        nonlocal newly_completed
        raw_rows.append(result)
        newly_completed += 1
        completed_count = len(raw_rows)
        print(
            f"[{completed_count:04d}/{total_runs:04d}] completed "
            f"{result['study_name']} | {result['topology']} | share={result['leader_share']:.2%} | "
            f"mode={result['leader_mode']} | perturbation={result['perturbation_id']} | seed={result['seed']}"
        )
        if newly_completed % checkpoint_every == 0:
            _order_raw_results(pd.DataFrame(raw_rows), grid_df).to_csv(checkpoint_path, index=False)

    conditions = [row.to_dict() for _, row in pending_df.iterrows()]
    workers = max(1, min(int(args.workers), os.cpu_count() or 1))
    if workers == 1:
        for condition in conditions:
            record_result(run_condition(condition))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_condition, condition): condition for condition in conditions}
            for future in as_completed(futures):
                record_result(future.result())

    raw_df = _order_raw_results(pd.DataFrame(raw_rows), grid_df)
    raw_df.to_csv(checkpoint_path, index=False)
    matched_effects_df = compute_matched_effects(raw_df)
    summary_df = aggregate_effects(matched_effects_df)
    control_summary_df = aggregate_controls(raw_df)
    checks_df = build_robustness_checks(summary_df)
    paths = export_outputs(
        output_dir,
        grid_df,
        raw_df,
        matched_effects_df,
        summary_df,
        control_summary_df,
        checks_df,
        profile_name=args.profile,
        study_name=args.study,
        notes=notes,
    )
    print(f"Saved sensitivity outputs to: {output_dir}")
    print(f"Matched effects: {paths['matched_effects']}")
    print(f"Robustness checks: {paths['checks']}")
    print(f"Resume checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
