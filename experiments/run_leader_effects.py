from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.base import (
    aggregate_raw_results,
    build_leader_effects_grid,
    export_leader_effects_outputs,
    extract_condition_result,
    run_single_condition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the leader-effects experiment matrix.")
    parser.add_argument(
        "--profile",
        default="main",
        choices=["trial", "main", "extended"],
        help="Experiment profile controlling seeds and default T_rounds.",
    )
    parser.add_argument(
        "--scenario",
        default="core",
        choices=["core", "no_leader_control", "robustness_t_rounds"],
        help="Experiment scenario: main paper matrix or appendix robustness check.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory. Defaults to outputs/leader_effects_main or outputs/leader_effects_extended.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for smoke-testing a subset of the grid.",
    )
    return parser.parse_args()


def default_output_dir(scenario_name: str) -> Path:
    if scenario_name == "core":
        return PROJECT_ROOT / "outputs" / "leader_effects_main"
    if scenario_name == "no_leader_control":
        return PROJECT_ROOT / "outputs" / "leader_effects_no_leader_control"
    return PROJECT_ROOT / "outputs" / "leader_effects_extended"


def main():
    args = parse_args()
    grid_df = build_leader_effects_grid(profile_name=args.profile, scenario_name=args.scenario)

    if args.max_runs is not None:
        grid_df = grid_df.head(int(args.max_runs)).copy()

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.scenario)

    raw_rows = []
    total_runs = len(grid_df)

    for run_index, (_, row) in enumerate(grid_df.iterrows(), start=1):
        condition = row.to_dict()
        print(
            f"[{run_index:04d}/{total_runs:04d}] "
            f"N={condition['N']} | topo={condition['topology']} | share={condition['leader_share']:.2%} | "
            f"mode={condition['leader_mode']} | T={condition['T_rounds']} | seed={condition['seed']}"
        )
        results = run_single_condition(condition)
        raw_rows.append(extract_condition_result(condition, results))

    raw_df = pd.DataFrame(raw_rows)
    summary_df = aggregate_raw_results(raw_df)

    notes = None
    if args.max_runs is not None:
        notes = f"Run truncated with --max-runs={args.max_runs} for smoke testing."

    exported_paths = export_leader_effects_outputs(
        output_root=output_dir,
        profile_name=args.profile,
        scenario_name=args.scenario,
        grid_df=grid_df,
        raw_df=raw_df,
        summary_df=summary_df,
        notes=notes,
    )

    print(f"Saved leader-effects outputs to: {output_dir}")
    print(f"Manifest: {exported_paths['manifest']}")
    print(f"Raw results: {exported_paths['raw_results']}")
    print(f"Summary results: {exported_paths['summary_results']}")


if __name__ == "__main__":
    main()
