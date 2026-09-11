"""Run the exploratory null experiment and retain ignored local outputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from opinion_model.scenarios.null import SHARED_FRAMEWORK_SOURCE_REVISION
from opinion_model.scenarios.null.config import load_null_experiment_config
from opinion_model.scenarios.null.experiment import (
    final_condition_metrics,
    run_null_condition,
    run_null_experiment,
    summarize_final_conditions,
)
from opinion_model.scenarios.null.observation import null_diagnostic_frames


def _git_state() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision, dirty


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/null.toml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/null"),
    )
    parser.add_argument(
        "--diagnostic-rounds",
        type=int,
        default=5,
        help="Number of initial rounds retained as detailed diagnostic tables.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    experiment = load_null_experiment_config(args.config)
    if not 1 <= args.diagnostic_rounds <= experiment.null.simulation.rounds:
        raise ValueError(
            "diagnostic-rounds must lie between one and the configured rounds."
        )

    started_at = datetime.now(timezone.utc)
    round_metrics = run_null_experiment(experiment)
    final_metrics = final_condition_metrics(round_metrics)
    summary = summarize_final_conditions(final_metrics)

    diagnostic_config = replace(
        experiment.null,
        simulation=replace(
            experiment.null.simulation,
            seed=experiment.seeds[0],
            rounds=args.diagnostic_rounds,
        ),
    )
    diagnostic_run = run_null_condition(
        diagnostic_config,
        extremism_threshold=experiment.extremism_threshold,
    )
    diagnostic_frames = null_diagnostic_frames(
        diagnostic_run.simulation_result,
        last_round=args.diagnostic_rounds,
    )
    completed_at = datetime.now(timezone.utc)

    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    output_directory = args.output_root / stamp
    output_directory.mkdir(parents=True, exist_ok=False)
    round_metrics.to_csv(output_directory / "round_metrics.csv", index=False)
    final_metrics.to_csv(output_directory / "final_metrics.csv", index=False)
    summary.to_csv(output_directory / "final_summary.csv", index=False)
    diagnostic_directory = output_directory / (
        f"diagnostic_seed_{experiment.seeds[0]}_rounds_1_"
        f"{args.diagnostic_rounds}"
    )
    diagnostic_directory.mkdir()
    for name, frame in diagnostic_frames.items():
        frame.to_csv(diagnostic_directory / f"{name}.csv", index=False)

    revision, dirty = _git_state()
    manifest = {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "source_config": str(args.config.resolve()),
        "shared_framework_source_revision": SHARED_FRAMEWORK_SOURCE_REVISION,
        "scenario_branch": "null",
        "scenario_revision": revision,
        "git_worktree_dirty": dirty,
        "diagnostic_seed": experiment.seeds[0],
        "diagnostic_rounds": args.diagnostic_rounds,
        "resolved_experiment": asdict(experiment),
    }
    with (output_directory / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print(f"Output directory: {output_directory.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
