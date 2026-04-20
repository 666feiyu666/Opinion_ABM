from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_SEED
from experiments.base import run_baseline_experiment, save_baseline_outputs
from metrics import format_round_summary


def run_baseline():
    results = run_baseline_experiment(seed=DEFAULT_SEED)

    for _, row in results["history_df"].iterrows():
        print(format_round_summary(int(row["round"]), row.to_dict()))

    output_paths = save_baseline_outputs(results, output_dir=PROJECT_ROOT / "outputs" / "baseline")
    print(f"Saved outputs to: {output_paths['history'].parent}")
    return results


if __name__ == "__main__":
    run_baseline()
