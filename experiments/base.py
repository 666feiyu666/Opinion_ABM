from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_SEED, make_notebook_baseline_params, make_simple_leader_influence_params
from main import run_simulation
from metrics import summarize_metric_distribution
from utils import ensure_directory, project_path
from visualization import (
    plot_final_opinion_distribution,
    plot_leader_effects_content_balance,
    plot_leader_effects_extremism,
    plot_leader_effects_heatmap,
    plot_leader_effects_mode_comparison,
    plot_leader_effects_overview,
    plot_no_leader_control_content,
    plot_no_leader_control_extremism,
    plot_no_leader_control_mean_opinion,
    plot_network_and_homophily,
    plot_opinion_leaders,
    plot_time_series_summaries,
    prepare_graph_for_visualization,
)

CORE_POPULATION_SIZES = [500, 1000, 1500]
CORE_TOPOLOGIES = ["BA", "ER", "WS", "SBM"]
CORE_LEADER_MODES = ["balanced", "positive", "negative"]
CORE_LEADER_SHARES = [0.01, 0.03, 0.05]

PROFILE_SPECS = {
    "trial": {
        "profile_name": "trial",
        "description": "Lightweight smoke-test profile for fast end-to-end checks.",
        "seed_count": 2,
        "T_rounds": 25,
    },
    "main": {
        "profile_name": "main",
        "description": "Core thesis-ready matrix on the tuned simple leader influence baseline.",
        "seed_count": 6,
        "T_rounds": 50,
    },
    "extended": {
        "profile_name": "extended",
        "description": "Appendix profile with extra seeds for tighter uncertainty intervals.",
        "seed_count": 10,
        "T_rounds": 50,
    },
}

SCENARIO_SPECS = {
    "core": {
        "scenario_name": "core",
        "description": "Main paper matrix over population size, topology, leader mode, and leader share.",
    },
    "no_leader_control": {
        "scenario_name": "no_leader_control",
        "description": "Control experiment with no opinion leaders and all other tuned parameters held fixed.",
        "population_sizes": CORE_POPULATION_SIZES,
        "topologies": CORE_TOPOLOGIES,
        "leader_modes": ["none"],
        "leader_shares": [0.0],
        "leader_selection_method": "none",
    },
    "robustness_t_rounds": {
        "scenario_name": "robustness_t_rounds",
        "description": "Appendix robustness check that varies T_rounds around the main baseline.",
        "population_sizes": [1000],
        "topologies": ["BA", "SBM"],
        "leader_modes": CORE_LEADER_MODES,
        "leader_shares": [0.01, 0.05],
        "T_rounds": [35, 50, 65],
        "seed_cap": 4,
    },
}

RAW_METRIC_COLUMNS = [
    "final_mean_opinion",
    "final_mean_abs_opinion",
    "final_std_opinion",
    "final_opinion_variance",
    "extremist_ratio",
    "moderate_ratio",
    "support_posts",
    "oppose_posts",
    "content_balance",
    "constructive_posts",
    "toxic_posts",
    "edge_count",
    "homophily_ratio",
    "cross_cutting_ratio",
    "sign_modularity",
    "actual_creators_total",
    "avg_exposure_size_mean",
]


def make_seed_list(seed_count: int, base_seed: int = DEFAULT_SEED) -> list[int]:
    return [int(base_seed + 17 * seed_index) for seed_index in range(seed_count)]


def get_profile_spec(profile_name: str = "main") -> dict:
    normalized_name = str(profile_name).strip().lower()
    if normalized_name not in PROFILE_SPECS:
        raise ValueError(f"Unsupported profile_name: {profile_name}")
    return dict(PROFILE_SPECS[normalized_name])


def get_scenario_spec(scenario_name: str = "core") -> dict:
    normalized_name = str(scenario_name).strip().lower()
    if normalized_name not in SCENARIO_SPECS:
        raise ValueError(f"Unsupported scenario_name: {scenario_name}")
    return dict(SCENARIO_SPECS[normalized_name])


def build_condition_id(condition: dict) -> str:
    share_label = f"{int(round(float(condition['leader_share']) * 100)):02d}pct"
    return (
        f"N{int(condition['N'])}_"
        f"{str(condition['topology']).upper()}_"
        f"{share_label}_"
        f"{str(condition['leader_mode']).lower()}_"
        f"T{int(condition['T_rounds'])}_"
        f"seed{int(condition['seed'])}"
    )


def build_leader_effects_grid(profile_name: str = "main", scenario_name: str = "core") -> pd.DataFrame:
    profile = get_profile_spec(profile_name)
    scenario = get_scenario_spec(scenario_name)

    if scenario["scenario_name"] == "core":
        population_sizes = CORE_POPULATION_SIZES
        topologies = CORE_TOPOLOGIES
        leader_modes = CORE_LEADER_MODES
        leader_shares = CORE_LEADER_SHARES
        rounds_list = [profile["T_rounds"]]
        seeds = make_seed_list(profile["seed_count"])
    else:
        population_sizes = scenario["population_sizes"]
        topologies = scenario["topologies"]
        leader_modes = scenario["leader_modes"]
        leader_shares = scenario["leader_shares"]
        rounds_list = scenario.get("T_rounds", [profile["T_rounds"]])
        seeds = make_seed_list(min(profile["seed_count"], int(scenario.get("seed_cap", profile["seed_count"]))))
    leader_selection_method = str(scenario.get("leader_selection_method", "top_in_degree"))

    rows = []
    for n_agents in population_sizes:
        for topology in topologies:
            for leader_share in leader_shares:
                for leader_mode in leader_modes:
                    for rounds in rounds_list:
                        for seed in seeds:
                            condition = {
                                "profile_name": profile["profile_name"],
                                "scenario_name": scenario["scenario_name"],
                                "N": int(n_agents),
                                "topology": str(topology).upper(),
                                "leader_share": float(leader_share),
                                "leader_mode": str(leader_mode).lower(),
                                "leader_selection_method": leader_selection_method,
                                "T_rounds": int(rounds),
                                "seed": int(seed),
                            }
                            condition["condition_id"] = build_condition_id(condition)
                            rows.append(condition)

    return pd.DataFrame(rows)


def build_simulation_params(condition: dict, extra_overrides: dict | None = None) -> dict:
    overrides = {
        "N": int(condition["N"]),
        "T_rounds": int(condition["T_rounds"]),
        "network_topology": str(condition["topology"]).upper(),
        "leader_share": float(condition["leader_share"]),
        "leader_selection_method": str(condition.get("leader_selection_method", "top_in_degree")),
    }
    if extra_overrides:
        overrides.update(extra_overrides)
    return make_simple_leader_influence_params(
        leader_opinion_mode=str(condition["leader_mode"]).lower(),
        overrides=overrides,
    )


def run_single_condition(
    condition: dict,
    *,
    track_opinions: bool = False,
    extra_param_overrides: dict | None = None,
):
    params = build_simulation_params(condition, extra_overrides=extra_param_overrides)
    return run_simulation(
        params=params,
        seed=int(condition["seed"]),
        rounds=int(condition["T_rounds"]),
        track_opinions=track_opinions,
    )


def extract_condition_result(condition: dict, results: dict) -> dict:
    history_df = results["history_df"].copy()
    final_state = results["final_state"]
    initialization_metadata = results.get("initialization_metadata", {})

    row = dict(condition)
    row.update(
        {
            "leader_count": int(results["agents"]["L"].sum()),
            "realized_leader_share": float(results["agents"]["L"].mean()),
            "initial_directed_edge_count": int(results["G_initial_undirected"].number_of_edges()),
            "final_mean_opinion": float(final_state["final_mean_opinion"]),
            "final_mean_abs_opinion": float(final_state["final_mean_abs_opinion"]),
            "final_std_opinion": float(final_state["final_std_opinion"]),
            "final_opinion_variance": float(final_state["final_opinion_variance"]),
            "extremist_ratio": float(final_state["final_extremist_ratio"]),
            "moderate_ratio": float(final_state["final_moderate_ratio"]),
            "edge_count": int(final_state["final_edge_count"]),
            "homophily_ratio": float(final_state["final_homophily_ratio"]),
            "cross_cutting_ratio": float(final_state["final_cross_cutting_ratio"]),
            "sign_modularity": float(final_state["final_sign_modularity"]),
            "support_posts": int(history_df["support_posts"].sum()),
            "oppose_posts": int(history_df["oppose_posts"].sum()),
            "constructive_posts": int(history_df["constructive_posts"].sum()),
            "toxic_posts": int(history_df["toxic_posts"].sum()),
            "actual_creators_total": int(history_df["actual_creators"].sum()),
            "avg_exposure_size_mean": float(history_df["avg_exposure_size"].mean()),
            "content_balance": int(history_df["support_posts"].sum() - history_df["oppose_posts"].sum()),
            "topology_target_average_degree": float(initialization_metadata.get("target_average_degree", float("nan"))),
            "network_metadata": json.dumps(initialization_metadata, ensure_ascii=True, sort_keys=True),
        }
    )
    return row


def aggregate_raw_results(raw_df: pd.DataFrame, metric_columns: list[str] | None = None) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    metric_columns = metric_columns or RAW_METRIC_COLUMNS
    group_columns = [
        "profile_name",
        "scenario_name",
        "N",
        "topology",
        "leader_share",
        "leader_mode",
        "leader_selection_method",
        "T_rounds",
    ]

    summary_rows = []
    for keys, group_df in raw_df.groupby(group_columns, dropna=False):
        summary_row = dict(zip(group_columns, keys))
        for metric_name in metric_columns:
            stats = summarize_metric_distribution(group_df[metric_name].to_numpy())
            summary_row[f"{metric_name}_mean"] = stats["mean"]
            summary_row[f"{metric_name}_std"] = stats["std"]
            summary_row[f"{metric_name}_sem"] = stats["sem"]
            summary_row[f"{metric_name}_ci_low"] = stats["ci_low"]
            summary_row[f"{metric_name}_ci_high"] = stats["ci_high"]
            summary_row[f"{metric_name}_n"] = stats["n"]
        summary_rows.append(summary_row)

    return pd.DataFrame(summary_rows).sort_values(group_columns).reset_index(drop=True)


def build_manifest(
    *,
    profile_name: str,
    scenario_name: str,
    grid_df: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    notes: str | None = None,
) -> dict:
    profile = get_profile_spec(profile_name)
    scenario = get_scenario_spec(scenario_name)
    leader_rule = "Exogenous leader share, then select top-k nodes by realized in-degree rank."
    if scenario["scenario_name"] == "no_leader_control":
        leader_rule = "No opinion leaders: leader_share=0 and leader_selection_method='none'."

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_name": profile["profile_name"],
        "profile_description": profile["description"],
        "scenario_name": scenario["scenario_name"],
        "scenario_description": scenario["description"],
        "output_dir": str(output_dir) if output_dir is not None else None,
        "condition_count": int(len(grid_df)),
        "completed_run_count": int(len(raw_df)) if raw_df is not None else 0,
        "population_sizes": sorted(grid_df["N"].unique().tolist()),
        "topologies": sorted(grid_df["topology"].unique().tolist()),
        "leader_modes": sorted(grid_df["leader_mode"].unique().tolist()),
        "leader_shares": sorted(grid_df["leader_share"].unique().tolist()),
        "rounds": sorted(grid_df["T_rounds"].unique().tolist()),
        "seeds": sorted(grid_df["seed"].unique().tolist()),
        "leader_rule": leader_rule,
        "fixed_parameter_base": "simple leader influence baseline",
        "default_main_rounds": PROFILE_SPECS["main"]["T_rounds"],
        "trial_rounds": PROFILE_SPECS["trial"]["T_rounds"],
        "notes": notes,
    }
    return manifest


def ensure_output_layout(output_root: str | Path) -> dict[str, Path]:
    base_dir = ensure_directory(output_root)
    layout = {
        "root": base_dir,
        "raw": ensure_directory(base_dir / "raw"),
        "summary": ensure_directory(base_dir / "summary"),
        "figures": ensure_directory(base_dir / "figures"),
        "tables": ensure_directory(base_dir / "tables"),
        "analysis_ready": ensure_directory(base_dir / "analysis_ready"),
    }
    return layout


def export_manifest(manifest: dict, output_root: str | Path) -> Path:
    manifest_path = ensure_directory(output_root) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest_path


def export_paper_tables(summary_df: pd.DataFrame, output_root: str | Path) -> dict[str, Path]:
    tables_dir = ensure_directory(Path(output_root) / "tables")

    key_columns = [
        "N",
        "topology",
        "leader_share",
        "leader_mode",
        "T_rounds",
        "final_mean_opinion_mean",
        "final_mean_opinion_ci_low",
        "final_mean_opinion_ci_high",
        "final_mean_abs_opinion_mean",
        "extremist_ratio_mean",
        "moderate_ratio_mean",
        "content_balance_mean",
        "constructive_posts_mean",
        "toxic_posts_mean",
        "homophily_ratio_mean",
        "sign_modularity_mean",
    ]
    core_table = summary_df[key_columns].copy().sort_values(["N", "topology", "leader_share", "leader_mode"])
    core_table_path = tables_dir / "paper_core_metrics.csv"
    core_table.to_csv(core_table_path, index=False)

    benchmark_table = (
        summary_df[summary_df["leader_share"].round(4) == 0.03]
        .copy()
        .sort_values(["N", "topology", "leader_mode"])
    )
    benchmark_table_path = tables_dir / "paper_benchmark_share_3pct.csv"
    benchmark_table.to_csv(benchmark_table_path, index=False)

    return {
        "paper_core_metrics": core_table_path,
        "paper_benchmark_share_3pct": benchmark_table_path,
    }


def export_analysis_ready_files(
    grid_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_root: str | Path,
) -> dict[str, Path]:
    analysis_dir = ensure_directory(Path(output_root) / "analysis_ready")
    outputs = {
        "experiment_grid": analysis_dir / "experiment_grid_analysis_ready.csv",
        "raw_results": analysis_dir / "raw_results_analysis_ready.csv",
        "summary_results": analysis_dir / "summary_results_analysis_ready.csv",
    }
    grid_df.to_csv(outputs["experiment_grid"], index=False)
    raw_df.to_csv(outputs["raw_results"], index=False)
    summary_df.to_csv(outputs["summary_results"], index=False)
    return outputs


def save_leader_effects_figures(
    summary_df: pd.DataFrame,
    output_root: str | Path,
    *,
    scenario_name: str = "core",
) -> dict[str, Path]:
    figures_dir = ensure_directory(Path(output_root) / "figures")
    saved_paths: dict[str, Path] = {}

    if scenario_name == "no_leader_control":
        figure_builders = [
            ("no_leader_mean_opinion", plot_no_leader_control_mean_opinion),
            ("no_leader_content_balance", plot_no_leader_control_content),
            ("no_leader_extremism", plot_no_leader_control_extremism),
        ]
    else:
        figure_builders = [
            ("leader_mode_final_mean_opinion", plot_leader_effects_mode_comparison),
            ("leader_share_content_balance", plot_leader_effects_content_balance),
            ("topology_extremist_ratio", plot_leader_effects_extremism),
            ("overview_dashboard", plot_leader_effects_overview),
            ("heatmap_final_mean_opinion_n1000", lambda df: plot_leader_effects_heatmap(df, metric="final_mean_opinion_mean", fixed_n=1000)),
        ]

    for stem, builder in figure_builders:
        fig, _ = builder(summary_df)
        target_path = figures_dir / f"{stem}.png"
        fig.savefig(target_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths[stem] = target_path

    return saved_paths


def export_leader_effects_outputs(
    *,
    output_root: str | Path,
    profile_name: str,
    scenario_name: str,
    grid_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    notes: str | None = None,
) -> dict[str, Path]:
    layout = ensure_output_layout(output_root)

    experiment_grid_path = layout["root"] / "experiment_grid.csv"
    raw_results_path = layout["raw"] / "raw_results.csv"
    summary_results_path = layout["summary"] / "summary_results.csv"

    grid_df.to_csv(experiment_grid_path, index=False)
    raw_df.to_csv(raw_results_path, index=False)
    summary_df.to_csv(summary_results_path, index=False)

    manifest = build_manifest(
        profile_name=profile_name,
        scenario_name=scenario_name,
        grid_df=grid_df,
        raw_df=raw_df,
        output_dir=layout["root"],
        notes=notes,
    )
    manifest_path = export_manifest(manifest, layout["root"])

    table_paths = export_paper_tables(summary_df, layout["root"])
    analysis_paths = export_analysis_ready_files(grid_df, raw_df, summary_df, layout["root"])
    figure_paths = save_leader_effects_figures(
        summary_df,
        layout["root"],
        scenario_name=scenario_name,
    )

    return {
        "manifest": manifest_path,
        "experiment_grid": experiment_grid_path,
        "raw_results": raw_results_path,
        "summary_results": summary_results_path,
        **table_paths,
        **analysis_paths,
        **figure_paths,
    }


def build_baseline_params(
    *,
    baseline_variant: str = "simple_leader_influence",
    leader_opinion_mode: str = "balanced",
    overrides: dict | None = None,
) -> dict:
    variant = str(baseline_variant).strip().lower()
    if variant == "simple_leader_influence":
        return make_simple_leader_influence_params(
            leader_opinion_mode=leader_opinion_mode,
            overrides=overrides,
        )
    if variant == "notebook_legacy":
        return make_notebook_baseline_params(overrides)
    raise ValueError(f"Unsupported baseline_variant: {baseline_variant}")


def run_baseline_experiment(
    *,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path | None = None,
    baseline_variant: str = "simple_leader_influence",
    leader_opinion_mode: str = "negative",
    overrides: dict | None = None,
) -> dict:
    params = build_baseline_params(
        baseline_variant=baseline_variant,
        leader_opinion_mode=leader_opinion_mode,
        overrides=overrides,
    )
    results = run_simulation(params=params, seed=seed)
    if output_dir is not None:
        save_baseline_outputs(results, output_dir=output_dir)
    return results


def save_baseline_outputs(results: dict, output_dir: str | Path | None = None) -> dict[str, Path]:
    params = results["params"]
    output_path = ensure_directory(output_dir or project_path("outputs", "baseline"))

    history_path = output_path / "history.csv"
    agents_path = output_path / "agents_final.csv"
    params_path = output_path / "params.json"
    history_path.unlink(missing_ok=True)
    agents_path.unlink(missing_ok=True)
    params_path.unlink(missing_ok=True)
    results["history_df"].to_csv(history_path, index=False)
    results["agents"].to_csv(agents_path, index=False)
    params_path.write_text(
        json.dumps(results["params"], indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )

    figure_paths = {
        "history": history_path,
        "agents": agents_path,
        "params": params_path,
    }

    fig, _ = plot_time_series_summaries(results["history_df"])
    figure_paths["time_series"] = output_path / "time_series.png"
    fig.savefig(figure_paths["time_series"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    graph_for_plot = prepare_graph_for_visualization(results["G"], results["agents"])

    fig, _ = plot_network_and_homophily(graph_for_plot, results["pos"])
    figure_paths["network_homophily"] = output_path / "network_homophily.png"
    fig.savefig(figure_paths["network_homophily"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_final_opinion_distribution(results["agents"], params=params)
    figure_paths["opinion_distribution"] = output_path / "opinion_distribution.png"
    fig.savefig(figure_paths["opinion_distribution"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_opinion_leaders(graph_for_plot, results["agents"], results["pos"])
    figure_paths["opinion_leaders"] = output_path / "opinion_leaders.png"
    fig.savefig(figure_paths["opinion_leaders"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    return figure_paths


def load_baseline_outputs(output_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    baseline_dir = Path(output_dir or project_path("outputs", "baseline"))
    return {
        "history_df": pd.read_csv(baseline_dir / "history.csv"),
        "agents_df": pd.read_csv(baseline_dir / "agents_final.csv"),
    }
