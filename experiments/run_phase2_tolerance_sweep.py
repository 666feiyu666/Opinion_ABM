from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_SEED
from experiments.run_phase1_tolerance_sweep import (  # noqa: F401
    MODERATE_BAND,
    SWEEP_THRESHOLDS,
    aggregate_sweep,
    build_seed_values,
    run_single_condition,
)
from experiments.run_phase1_tolerance_sweep import (
    merge_phase1_batches,
    phase1_overrides,
    run_phase1_experiment,
)


def phase2_overrides(rounds: int, tolerance_threshold: float) -> dict:
    return phase1_overrides(rounds=rounds, tolerance_threshold=tolerance_threshold)


def run_phase2_experiment(
    rounds: int = 50,
    seeds: int = 20,
    output_subdir: str = "phase2_tolerance_sweep",
    workers: int = 1,
    seed_start: int = DEFAULT_SEED,
):
    return run_phase1_experiment(
        rounds=rounds,
        seeds=seeds,
        output_subdir=output_subdir,
        workers=workers,
        seed_start=seed_start,
    )


def merge_phase2_batches(input_subdirs: list[str], output_subdir: str):
    return merge_phase1_batches(input_subdirs=input_subdirs, output_subdir=output_subdir)


def parse_args():
    parser = argparse.ArgumentParser(description="Compatibility wrapper for the old phase2 sweep entrypoint.")
    parser.add_argument("--rounds", type=int, default=50, help="Simulation rounds per run.")
    parser.add_argument("--seeds", type=int, default=20, help="Monte Carlo repetitions per threshold.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED,
        help="Starting random seed for the batch.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes.")
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="phase2_tolerance_sweep",
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
        merge_phase2_batches(
            input_subdirs=args.merge_input_subdirs,
            output_subdir=args.output_subdir,
        )
    else:
        run_phase2_experiment(
            rounds=args.rounds,
            seeds=args.seeds,
            output_subdir=args.output_subdir,
            workers=args.workers,
            seed_start=args.seed_start,
        )
