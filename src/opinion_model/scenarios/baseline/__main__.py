"""Run the exploratory integrated baseline and retain ignored local outputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from opinion_model.scenarios.baseline.config import (
    load_baseline_experiment_config,
)
from opinion_model.scenarios.baseline.experiment import (
    final_condition_metrics,
    run_baseline_experiment,
    summarize_final_conditions,
)


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
        default=Path("configs/baseline.toml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/baseline"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    experiment = load_baseline_experiment_config(args.config)
    started_at = datetime.now(timezone.utc)
    round_metrics = run_baseline_experiment(experiment)
    final_metrics = final_condition_metrics(round_metrics)
    summary = summarize_final_conditions(final_metrics)
    completed_at = datetime.now(timezone.utc)

    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    output_directory = args.output_root / stamp
    output_directory.mkdir(parents=True, exist_ok=False)
    round_metrics.to_csv(output_directory / "round_metrics.csv", index=False)
    final_metrics.to_csv(output_directory / "final_metrics.csv", index=False)
    summary.to_csv(output_directory / "final_summary.csv", index=False)

    revision, dirty = _git_state()
    manifest = {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "source_config": str(args.config.resolve()),
        "git_revision": revision,
        "git_worktree_dirty": dirty,
        "resolved_experiment": asdict(experiment),
    }
    with (output_directory / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print(f"Output directory: {output_directory.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
