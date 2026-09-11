"""Run the complete matched comparison and retain ignored local outputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from opinion_model.scenarios.comparison import (
    load_comparison_experiment_config,
    run_comparison_experiment,
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
        default=Path("configs/comparison.toml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/comparison"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    experiment = load_comparison_experiment_config(args.config)
    started_at = datetime.now(timezone.utc)
    result = run_comparison_experiment(experiment)
    completed_at = datetime.now(timezone.utc)

    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    output_directory = args.output_root / stamp
    output_directory.mkdir(parents=True, exist_ok=False)
    result.round_metrics.to_csv(
        output_directory / "round_metrics.csv",
        index=False,
    )
    result.final_metrics.to_csv(
        output_directory / "final_metrics.csv",
        index=False,
    )
    result.final_summary.to_csv(
        output_directory / "final_summary.csv",
        index=False,
    )
    scenario_directory = output_directory / "scenarios"
    scenario_directory.mkdir()
    for name, frame in result.scenario_round_metrics.items():
        frame.to_csv(scenario_directory / f"{name}_round_metrics.csv", index=False)

    revision, dirty = _git_state()
    resolved_scenarios = {
        "null": asdict(experiment.null),
        "opleader": asdict(experiment.opleader),
        "platform": asdict(experiment.platform),
        "baseline": asdict(experiment.baseline),
    }
    manifest = {
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "comparison_id": experiment.experiment_id,
        "status": experiment.status,
        "source_config": str(args.config.resolve()),
        "scenario_config_paths": {
            name: str(path) for name, path in experiment.source_paths.items()
        },
        "git_revision": revision,
        "git_worktree_dirty": dirty,
        "resolved_scenarios": resolved_scenarios,
        "comparison_contract": {
            "interest_decay": "shared across all four scenarios",
            "finite_attention": "active only in platform and baseline",
            "platform_contrast": (
                "joint effect of out-of-network availability, finite attention, "
                "and exposure-driven network adaptation"
            ),
            "opinion_leader_contrast": (
                "joint effect of leader initialization, origination advantage, "
                "and source-dependent evidence weight"
            ),
        },
    }
    with (output_directory / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print(f"Output directory: {output_directory.resolve()}")
    print(result.final_summary.to_string(index=False))


if __name__ == "__main__":
    main()
