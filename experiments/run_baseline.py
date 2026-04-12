from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEFAULT_SEED, make_notebook_baseline_params
from main import run_simulation
from metrics import format_round_summary
from utils import ensure_directory, project_path
from visualization import (
    plot_final_opinion_distribution,
    plot_network_and_homophily,
    plot_opinion_leaders,
    plot_time_series_summaries,
)


def run_baseline():
    params = make_notebook_baseline_params()
    results = run_simulation(params=params, seed=DEFAULT_SEED)

    for _, row in results["history_df"].iterrows():
        print(format_round_summary(int(row["round"]), row.to_dict()))

    output_dir = ensure_directory(project_path("outputs", "baseline"))
    history_path = output_dir / "history.csv"
    agents_path = output_dir / "agents_final.csv"

    results["history_df"].to_csv(history_path, index=False)
    results["agents"].to_csv(agents_path, index=False)

    fig, _ = plot_time_series_summaries(results["history_df"])
    fig.savefig(output_dir / "time_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_network_and_homophily(results["G_updated"], results["pos"])
    fig.savefig(output_dir / "network_homophily.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_final_opinion_distribution(results["agents"], params=params)
    fig.savefig(output_dir / "opinion_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_opinion_leaders(results["G_updated"], results["agents"], results["pos"])
    fig.savefig(output_dir / "opinion_leaders.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved outputs to: {output_dir}")
    return results


if __name__ == "__main__":
    run_baseline()
