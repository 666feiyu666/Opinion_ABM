from __future__ import annotations

import argparse
import os
import sys
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
from initialization import initialize_model
from main import run_one_round
from metrics import format_round_summary
from utils import ensure_directory, project_path, set_random_seed


DEFAULT_TOLERANCE_THRESHOLD = 0.3
DEFAULT_LEADER_WEIGHT = 20.0
DEFAULT_OUTPUT_SUBDIR = "phase2_leader_volume_overload_population"


def phase2_overrides(rounds: int, leader_weight: float, tolerance_threshold: float) -> dict:
    return {
        "T_rounds": int(rounds),
        "tolerance_threshold": float(tolerance_threshold),
        "w_l": float(leader_weight),
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def build_zone_record(
    round_number: int,
    agents: pd.DataFrame,
    params: dict,
    initial_in_zone_nonleaders: set[int],
) -> dict:
    theta_t = float(params["tolerance_threshold"])
    in_zone_mask = agents["o_t"].abs() < theta_t
    nonleader_mask = agents["L"] == 0
    leader_mask = ~nonleader_mask

    current_nonleader_in_zone_nodes = set(
        agents.loc[in_zone_mask & nonleader_mask, "node"].astype(int).tolist()
    )
    initial_survivors = initial_in_zone_nonleaders & current_nonleader_in_zone_nodes
    escaped_from_initial = initial_in_zone_nonleaders - current_nonleader_in_zone_nodes
    reentered_from_initial = {
        node
        for node in initial_in_zone_nonleaders
        if node in current_nonleader_in_zone_nodes and round_number > 0
    }

    total_in_zone = int(in_zone_mask.sum())
    total_nonleaders = int(nonleader_mask.sum())
    total_leaders = int(leader_mask.sum())
    nonleader_in_zone = int((in_zone_mask & nonleader_mask).sum())
    leader_in_zone = int((in_zone_mask & leader_mask).sum())

    return {
        "round": int(round_number),
        "tolerance_threshold": theta_t,
        "total_agents": int(len(agents)),
        "total_leaders": total_leaders,
        "total_nonleaders": total_nonleaders,
        "in_zone_total_count": total_in_zone,
        "in_zone_total_ratio": _ratio(total_in_zone, len(agents)),
        "in_zone_nonleader_count": nonleader_in_zone,
        "in_zone_nonleader_ratio": _ratio(nonleader_in_zone, total_nonleaders),
        "in_zone_leader_count": leader_in_zone,
        "in_zone_leader_ratio": _ratio(leader_in_zone, total_leaders),
        "out_zone_nonleader_count": int(total_nonleaders - nonleader_in_zone),
        "out_zone_nonleader_ratio": _ratio(total_nonleaders - nonleader_in_zone, total_nonleaders),
        "initial_in_zone_nonleader_count": int(len(initial_in_zone_nonleaders)),
        "initial_in_zone_survivor_count": int(len(initial_survivors)),
        "initial_in_zone_survivor_ratio": _ratio(len(initial_survivors), len(initial_in_zone_nonleaders)),
        "initial_in_zone_escape_count": int(len(escaped_from_initial)),
        "initial_in_zone_escape_ratio": _ratio(len(escaped_from_initial), len(initial_in_zone_nonleaders)),
        "initial_in_zone_reentered_count": int(len(reentered_from_initial)),
        "mean_opinion_in_zone_nonleaders": float(
            agents.loc[in_zone_mask & nonleader_mask, "o_t"].mean()
        ) if nonleader_in_zone > 0 else 0.0,
        "mean_tau_in_zone_nonleaders": float(
            agents.loc[in_zone_mask & nonleader_mask, "tau_t"].mean()
        ) if nonleader_in_zone > 0 else 0.0,
        "mean_tau_out_zone_nonleaders": float(
            agents.loc[(~in_zone_mask) & nonleader_mask, "tau_t"].mean()
        ) if nonleader_in_zone < total_nonleaders else 0.0,
    }


def summarize_zone_shift(zone_df: pd.DataFrame) -> pd.DataFrame:
    if zone_df.empty:
        return pd.DataFrame()

    start = zone_df.iloc[0]
    end = zone_df.iloc[-1]
    summary = {
        "rounds": int(end["round"]),
        "theta_T": float(end["tolerance_threshold"]),
        "initial_in_zone_nonleader_count": int(start["in_zone_nonleader_count"]),
        "final_in_zone_nonleader_count": int(end["in_zone_nonleader_count"]),
        "delta_in_zone_nonleader_count": int(
            end["in_zone_nonleader_count"] - start["in_zone_nonleader_count"]
        ),
        "initial_in_zone_nonleader_ratio": float(start["in_zone_nonleader_ratio"]),
        "final_in_zone_nonleader_ratio": float(end["in_zone_nonleader_ratio"]),
        "delta_in_zone_nonleader_ratio": float(
            end["in_zone_nonleader_ratio"] - start["in_zone_nonleader_ratio"]
        ),
        "initial_survivor_count": int(start["initial_in_zone_survivor_count"]),
        "final_survivor_count": int(end["initial_in_zone_survivor_count"]),
        "final_survivor_ratio": float(end["initial_in_zone_survivor_ratio"]),
        "final_escape_count": int(end["initial_in_zone_escape_count"]),
        "final_escape_ratio": float(end["initial_in_zone_escape_ratio"]),
        "min_in_zone_nonleader_count": int(zone_df["in_zone_nonleader_count"].min()),
        "min_in_zone_nonleader_round": int(
            zone_df.loc[zone_df["in_zone_nonleader_count"].idxmin(), "round"]
        ),
        "max_escape_ratio": float(zone_df["initial_in_zone_escape_ratio"].max()),
        "max_escape_ratio_round": int(
            zone_df.loc[zone_df["initial_in_zone_escape_ratio"].idxmax(), "round"]
        ),
    }
    return pd.DataFrame([summary])


def plot_zone_counts(zone_df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(
        zone_df["round"],
        zone_df["in_zone_nonleader_count"],
        marker="o",
        linewidth=2.0,
        color="#2563eb",
        label="Non-leaders inside tolerance",
    )
    axes[0].plot(
        zone_df["round"],
        zone_df["initial_in_zone_survivor_count"],
        marker="s",
        linewidth=1.8,
        color="#059669",
        label="Initial moderates still inside",
    )
    axes[0].set_ylabel("Count")
    axes[0].set_title("Tolerance-Zone Population Under Leader Volume Overload")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best")

    axes[1].plot(
        zone_df["round"],
        zone_df["in_zone_nonleader_ratio"],
        marker="o",
        linewidth=2.0,
        color="#2563eb",
        label="Non-leader in-zone ratio",
    )
    axes[1].plot(
        zone_df["round"],
        zone_df["initial_in_zone_escape_ratio"],
        marker="s",
        linewidth=1.8,
        color="#dc2626",
        label="Initial moderate escape ratio",
    )
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Ratio")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_zone_confidence(zone_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        zone_df["round"],
        zone_df["mean_tau_in_zone_nonleaders"],
        marker="o",
        linewidth=2.0,
        color="#7c3aed",
        label="Mean tau inside zone",
    )
    ax.plot(
        zone_df["round"],
        zone_df["mean_tau_out_zone_nonleaders"],
        marker="s",
        linewidth=1.8,
        color="#ea580c",
        label="Mean tau outside zone",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean tau")
    ax.set_title("Confidence Split Between In-Zone and Out-of-Zone Non-Leaders")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_macro_summary(history_df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    plot_columns = [
        ("avg_exposure_size", "Avg Exposure Size"),
        ("extremist_ratio", "Extremist Ratio"),
        ("moderate_ratio", "Moderate Ratio"),
        ("mean_abs_opinion", "Mean Absolute Opinion"),
    ]

    for ax, (column, title) in zip(axes, plot_columns):
        ax.plot(history_df["round"], history_df[column], marker="o", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_leader_volume_overload(
    rounds: int = 50,
    seed: int = DEFAULT_SEED,
    leader_weight: float = DEFAULT_LEADER_WEIGHT,
    tolerance_threshold: float = DEFAULT_TOLERANCE_THRESHOLD,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
):
    params = make_notebook_baseline_params(
        phase2_overrides(
            rounds=rounds,
            leader_weight=leader_weight,
            tolerance_threshold=tolerance_threshold,
        )
    )

    rng = set_random_seed(seed)
    graph, _, agents, blocks, _ = initialize_model(params, seed=seed)

    theta_t = float(params["tolerance_threshold"])
    initial_in_zone_nonleaders = set(
        agents.loc[(agents["L"] == 0) & (agents["o_t"].abs() < theta_t), "node"]
        .astype(int)
        .tolist()
    )

    zone_records = [
        build_zone_record(
            round_number=0,
            agents=agents,
            params=params,
            initial_in_zone_nonleaders=initial_in_zone_nonleaders,
        )
    ]
    round_records = []

    for round_number in range(1, params["T_rounds"] + 1):
        graph, agents, posts, exposure_sets, summary = run_one_round(
            graph,
            agents,
            blocks,
            params,
            rng,
            current_round=round_number,
        )
        summary["round"] = round_number
        round_records.append(summary)
        print(format_round_summary(round_number, summary))

        zone_records.append(
            build_zone_record(
                round_number=round_number,
                agents=agents,
                params=params,
                initial_in_zone_nonleaders=initial_in_zone_nonleaders,
            )
        )

    history_df = pd.DataFrame(round_records)
    zone_df = pd.DataFrame(zone_records).sort_values("round").reset_index(drop=True)
    summary_df = summarize_zone_shift(zone_df)

    output_dir = ensure_directory(project_path("outputs", output_subdir))
    history_path = output_dir / "history.csv"
    zone_path = output_dir / "tolerance_zone_counts.csv"
    summary_path = output_dir / "tolerance_zone_summary.csv"
    manifest_path = output_dir / "experiment_manifest.csv"

    history_df.to_csv(history_path, index=False)
    zone_df.to_csv(zone_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    manifest_df = pd.DataFrame(
        [
            {
                "phase": "phase2_leader_volume_overload",
                "seed": int(seed),
                "rounds": int(rounds),
                "tolerance_threshold": float(tolerance_threshold),
                "leader_weight": float(leader_weight),
                "baseline": "make_notebook_baseline_params",
            }
        ]
    )
    manifest_df.to_csv(manifest_path, index=False)

    plot_zone_counts(zone_df=zone_df, output_path=output_dir / "tolerance_zone_counts.png")
    plot_zone_confidence(
        zone_df=zone_df,
        output_path=output_dir / "tolerance_zone_confidence.png",
    )
    plot_macro_summary(history_df=history_df, output_path=output_dir / "macro_summary.png")

    if not summary_df.empty:
        print("\nTolerance-zone shift summary:")
        print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved Phase 2.0 leader overload outputs to: {output_dir}")

    return {
        "params": params,
        "history_df": history_df,
        "zone_df": zone_df,
        "summary_df": summary_df,
        "output_dir": output_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Phase 2.0 leader volume overload sandbox test."
    )
    parser.add_argument("--rounds", type=int, default=50, help="Simulation rounds.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--leader-weight",
        type=float,
        default=DEFAULT_LEADER_WEIGHT,
        help="Exposure weight assigned to leader posts.",
    )
    parser.add_argument(
        "--tolerance-threshold",
        type=float,
        default=DEFAULT_TOLERANCE_THRESHOLD,
        help="Tolerance threshold theta_T.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Subdirectory under outputs/ for CSV and figure outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_leader_volume_overload(
        rounds=args.rounds,
        seed=args.seed,
        leader_weight=args.leader_weight,
        tolerance_threshold=args.tolerance_threshold,
        output_subdir=args.output_subdir,
    )
