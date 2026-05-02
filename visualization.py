from __future__ import annotations

import os
from pathlib import Path
import json

DEFAULT_MPL_CACHE = Path(__file__).resolve().parent / ".mpl-cache"
DEFAULT_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_CACHE))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from metrics import compute_average_neighbor_opinions


def plot_time_series_summaries(history_df):
    series_spec = [
        ("actual_creators", "Number of Content Creators by Round", "Creators"),
        ("avg_exposure_size", "Average Exposure Size by Round", "Average Exposure Size"),
        ("mean_opinion", "Mean Latent Opinion by Round", "Mean Opinion"),
        ("edge_count", "Directed Edge Count by Round", "Number of Edges"),
    ]
    if {"mean_involvement", "involved_ratio"}.issubset(history_df.columns):
        series_spec.extend(
            [
                ("mean_involvement", "Mean Involvement by Round", "Mean Involvement"),
                ("involved_ratio", "High-Involvement Ratio by Round", "Share of Agents"),
            ]
        )

    n_plots = len(series_spec)
    ncols = 3 if n_plots > 4 else 2
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (column, title, ylabel) in zip(axes, series_spec):
        ax.plot(history_df["round"], history_df[column], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


def prepare_graph_for_visualization(graph, agents):
    graph_updated = graph.copy()
    for node in graph_updated.nodes():
        graph_updated.nodes[node]["opinion"] = agents.at[node, "o_t"]
        graph_updated.nodes[node]["leader"] = agents.at[node, "L"]
    return graph_updated


def plot_network_and_homophily(graph, pos):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    opinions = [graph.nodes[n]["opinion"] for n in graph.nodes()]

    ax_net = axes[0]
    nodes_draw = nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=opinions,
        cmap=plt.cm.coolwarm,
        node_size=35,
        ax=ax_net,
        vmin=-1,
        vmax=1,
    )
    nx.draw_networkx_edges(graph, pos, alpha=0.05, ax=ax_net)
    ax_net.set_title("Network Graph by Agent Opinions\n(Red: Pro, Blue: Anti)")
    ax_net.axis("off")

    cbar = fig.colorbar(nodes_draw, ax=ax_net, orientation="horizontal", shrink=0.7, pad=0.05)
    cbar.set_label("Agent Opinion [-1, 1]")

    ax_density = axes[1]
    avg_neighbor_opinions = compute_average_neighbor_opinions(graph)
    sns.kdeplot(
        x=opinions,
        y=avg_neighbor_opinions,
        fill=True,
        cmap="mako",
        thresh=0.05,
        ax=ax_density,
    )
    ax_density.set_xlabel("Agents' opinions")
    ax_density.set_ylabel("Average of neighbours' opinions")
    ax_density.set_title("2D Density Plot of Opinion Homophily")
    ax_density.set_xlim(-1.1, 1.1)
    ax_density.set_ylim(-1.1, 1.1)

    fig.tight_layout()
    return fig, axes


def plot_final_opinion_distribution(agents, params: dict | None = None, bins: int = 25):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, bin_edges, _ = ax.hist(
        agents["o_t"],
        bins=bins,
        edgecolor="black",
        alpha=0.75,
        label="Final opinions",
    )

    if params is not None:
        initial_mean = float(params.get("opinion_mean", 0.0))
        initial_std = float(params.get("opinion_std", 0.0))
        if initial_std > 0:
            x_values = np.linspace(-1.0, 1.0, 400)
            bin_width = bin_edges[1] - bin_edges[0]
            density = (
                np.exp(-0.5 * ((x_values - initial_mean) / initial_std) ** 2)
                / (initial_std * np.sqrt(2.0 * np.pi))
            )
            scaled_density = len(agents) * bin_width * density
            ax.plot(
                x_values,
                scaled_density,
                color="#dc2626",
                linewidth=2.2,
                linestyle="--",
                label="Initial normal curve",
            )

    ax.set_title("Final Latent Opinion Distribution After Simulation")
    ax.set_xlabel("Latent Opinion")
    ax.set_ylabel("Number of Agents")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_opinion_leaders(graph, agents, pos):
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(
        graph,
        pos,
        alpha=0.05,
        width=0.4,
        edge_color="gray",
        ax=ax,
    )

    ordinary_nodes = agents[agents["L"] == 0]["node"].tolist()
    leader_nodes = agents[agents["L"] == 1]["node"].tolist()

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=ordinary_nodes,
        node_size=35,
        node_color="lightgray",
        edgecolors="black",
        linewidths=0.2,
        label="Ordinary Users",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=leader_nodes,
        node_size=180,
        node_color="gold",
        edgecolors="black",
        linewidths=0.8,
        label="Opinion Leaders",
        ax=ax,
    )

    ax.set_title("Opinion Leaders Highlighted in Final Network")
    ax.legend()
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def plot_baseline_validation(history_df, opinion_trajectory_df, graph, pos, sample_size: int = 80):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    ax_series = axes[0]
    trajectory_df = opinion_trajectory_df.copy()
    if not trajectory_df.empty:
        unique_nodes = sorted(trajectory_df["node"].unique())
        sampled_nodes = unique_nodes[:sample_size]
        if len(unique_nodes) > sample_size:
            sampled_nodes = list(np.linspace(0, len(unique_nodes) - 1, sample_size, dtype=int))
            sampled_nodes = [unique_nodes[index] for index in sampled_nodes]

        sampled_df = trajectory_df[trajectory_df["node"].isin(sampled_nodes)]
        for _, node_df in sampled_df.groupby("node"):
            ax_series.plot(
                node_df["round"],
                node_df["opinion"],
                color="#3b82f6",
                alpha=0.08,
                linewidth=0.9,
            )

        mean_by_round = trajectory_df.groupby("round", as_index=False)["opinion"].mean()
        ax_series.plot(
            mean_by_round["round"],
            mean_by_round["opinion"],
            color="#111827",
            linewidth=2.0,
            label="Mean opinion",
        )

    if not history_df.empty:
        ax_series_twin = ax_series.twinx()
        ax_series_twin.plot(
            history_df["round"],
            history_df["extremist_ratio"],
            color="#ef4444",
            linewidth=2.0,
            linestyle="--",
            label="Extremist ratio",
        )
        ax_series_twin.set_ylabel("Extremist ratio")
        ax_series_twin.set_ylim(0.0, 1.0)

    ax_series.axhline(0.7, color="gray", linestyle=":", linewidth=1.0)
    ax_series.axhline(-0.7, color="gray", linestyle=":", linewidth=1.0)
    ax_series.set_title("Baseline Polarization Path")
    ax_series.set_xlabel("Round")
    ax_series.set_ylabel("Opinion")
    ax_series.set_ylim(-1.05, 1.05)
    ax_series.grid(alpha=0.2)

    ax_network = axes[1]
    opinions = [graph.nodes[n]["opinion"] for n in graph.nodes()]
    nodes_draw = nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=opinions,
        cmap=plt.cm.coolwarm,
        node_size=38,
        ax=ax_network,
        vmin=-1,
        vmax=1,
    )
    nx.draw_networkx_edges(graph, pos, alpha=0.05, width=0.5, ax=ax_network)
    ax_network.set_title("Final Network Topology Under Low Acceptance Latitude")
    ax_network.axis("off")
    cbar = fig.colorbar(nodes_draw, ax=ax_network, orientation="horizontal", shrink=0.75, pad=0.03)
    cbar.set_label("Final opinion")

    fig.tight_layout()
    return fig, axes


def plot_healing_curves(sweep_summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    x_values = sweep_summary_df["tolerance_threshold"]

    ax_left = axes[0]
    ax_left.plot(
        x_values,
        sweep_summary_df["opinion_variance_mean"],
        color="#1d4ed8",
        marker="o",
        linewidth=2.2,
    )
    ax_left.fill_between(
        x_values,
        sweep_summary_df["opinion_variance_ci_low"],
        sweep_summary_df["opinion_variance_ci_high"],
        color="#93c5fd",
        alpha=0.25,
    )
    ax_left.set_xlabel("tolerance_threshold")
    ax_left.set_ylabel("Final opinion variance", color="#1d4ed8")
    ax_left.tick_params(axis="y", labelcolor="#1d4ed8")
    ax_left.grid(alpha=0.2)

    ax_right = ax_left.twinx()
    ax_right.plot(
        x_values,
        sweep_summary_df["extremist_ratio_mean"],
        color="#dc2626",
        marker="s",
        linewidth=2.2,
    )
    ax_right.fill_between(
        x_values,
        sweep_summary_df["extremist_ratio_ci_low"],
        sweep_summary_df["extremist_ratio_ci_high"],
        color="#fca5a5",
        alpha=0.22,
    )
    ax_right.set_ylabel("Extremist ratio", color="#dc2626")
    ax_right.tick_params(axis="y", labelcolor="#dc2626")
    ax_right.set_ylim(0.0, 1.0)
    ax_left.set_title("Healing Curve: Variance and Extremism")

    ax_homophily = axes[1]
    ax_homophily.plot(
        x_values,
        sweep_summary_df["homophily_ratio_mean"],
        color="#7c3aed",
        marker="o",
        linewidth=2.0,
        label="Homophily ratio",
    )
    ax_homophily.fill_between(
        x_values,
        sweep_summary_df["homophily_ratio_ci_low"],
        sweep_summary_df["homophily_ratio_ci_high"],
        color="#c4b5fd",
        alpha=0.22,
    )
    ax_homophily.plot(
        x_values,
        sweep_summary_df["cross_cutting_ratio_mean"],
        color="#059669",
        marker="^",
        linewidth=2.0,
        label="Cross-cutting ratio",
    )
    ax_homophily.fill_between(
        x_values,
        sweep_summary_df["cross_cutting_ratio_ci_low"],
        sweep_summary_df["cross_cutting_ratio_ci_high"],
        color="#a7f3d0",
        alpha=0.22,
    )
    ax_homophily.set_xlabel("tolerance_threshold")
    ax_homophily.set_ylabel("Network segregation")
    ax_homophily.set_title("Network Echo-Chamber Decay")
    ax_homophily.grid(alpha=0.2)
    ax_homophily.legend()

    fig.tight_layout()
    return fig, axes


def plot_tolerance_sweep_curves(sweep_summary_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    plot_spec = [
        (
            "extremist_ratio",
            "Extremist Ratio",
            "Final extremist ratio",
            "#dc2626",
            "#fca5a5",
            (0.0, 1.0),
        ),
        (
            "opinion_variance",
            "Opinion Variance",
            "Final opinion variance",
            "#2563eb",
            "#93c5fd",
            None,
        ),
        (
            "moderate_ratio",
            "Moderate Survival",
            "Final moderate ratio",
            "#059669",
            "#a7f3d0",
            (0.0, 1.0),
        ),
    ]

    x_values = sweep_summary_df["tolerance_threshold"]

    for ax, (metric_key, title, ylabel, line_color, fill_color, y_limits) in zip(axes, plot_spec):
        ax.plot(
            x_values,
            sweep_summary_df[f"{metric_key}_mean"],
            color=line_color,
            marker="o",
            linewidth=2.2,
        )
        ax.fill_between(
            x_values,
            sweep_summary_df[f"{metric_key}_ci_low"],
            sweep_summary_df[f"{metric_key}_ci_high"],
            color=fill_color,
            alpha=0.28,
        )
        ax.set_title(title)
        ax.set_xlabel("tolerance_threshold")
        ax.set_ylabel(ylabel)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig, axes


def plot_final_opinion_heatmap(opinion_samples_df, bins: int = 41):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    heatmap_df = opinion_samples_df.copy()
    heatmap_df["tolerance_threshold"] = heatmap_df["tolerance_threshold"].astype(float)

    bin_edges = np.linspace(-1.0, 1.0, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    thresholds = sorted(heatmap_df["tolerance_threshold"].unique())
    density_rows = []
    for threshold in thresholds:
        threshold_opinions = heatmap_df.loc[
            heatmap_df["tolerance_threshold"] == threshold,
            "opinion",
        ].to_numpy()
        counts, _ = np.histogram(threshold_opinions, bins=bin_edges, density=False)
        total = counts.sum()
        densities = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
        density_rows.append(densities)

    density_matrix = np.asarray(density_rows)
    sns.heatmap(
        density_matrix,
        cmap="mako",
        cbar_kws={"label": "Share of final opinions"},
        xticklabels=np.round(bin_centers, 2),
        yticklabels=[f"{threshold:.1f}" for threshold in thresholds],
        ax=ax,
    )
    ax.set_xlabel("Final opinion bin center")
    ax.set_ylabel("tolerance_threshold")
    ax.set_title("Final Opinion Distribution Across Tolerance Thresholds")

    xtick_positions = np.linspace(0, len(bin_centers) - 1, 9, dtype=int)
    ax.set_xticks(xtick_positions + 0.5)
    ax.set_xticklabels([f"{bin_centers[index]:.2f}" for index in xtick_positions], rotation=0)

    fig.tight_layout()
    return fig, ax


def load_tolerance_sweep_outputs(output_dir: str | Path) -> dict[str, pd.DataFrame]:
    base_dir = Path(output_dir)
    return {
        "sweep_raw_df": pd.read_csv(base_dir / "sweep_raw.csv"),
        "sweep_summary_df": pd.read_csv(base_dir / "sweep_summary.csv"),
        "opinion_samples_df": pd.read_csv(base_dir / "final_opinion_samples.csv"),
    }


def build_tolerance_sweep_notebook_table(sweep_summary_df: pd.DataFrame) -> pd.DataFrame:
    table = sweep_summary_df[
        [
            "tolerance_threshold",
            "extremist_ratio_mean",
            "extremist_ratio_ci_low",
            "extremist_ratio_ci_high",
            "opinion_variance_mean",
            "opinion_variance_ci_low",
            "opinion_variance_ci_high",
            "moderate_ratio_mean",
            "moderate_ratio_ci_low",
            "moderate_ratio_ci_high",
            "mean_abs_opinion_mean",
            "homophily_ratio_mean",
        ]
    ].copy()
    return table.rename(
        columns={
            "tolerance_threshold": "theta_T",
            "extremist_ratio_mean": "extremist_mean",
            "extremist_ratio_ci_low": "extremist_ci_low",
            "extremist_ratio_ci_high": "extremist_ci_high",
            "opinion_variance_mean": "variance_mean",
            "opinion_variance_ci_low": "variance_ci_low",
            "opinion_variance_ci_high": "variance_ci_high",
            "moderate_ratio_mean": "moderate_mean",
            "moderate_ratio_ci_low": "moderate_ci_low",
            "moderate_ratio_ci_high": "moderate_ci_high",
            "mean_abs_opinion_mean": "mean_abs_opinion",
            "homophily_ratio_mean": "homophily_mean",
        }
    )


def plot_pre_post_comparison(pathology_result, healing_result, pathology_threshold: float, healing_threshold: float):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    comparison_spec = [
        (0, pathology_result, pathology_threshold, "Pathological state"),
        (1, healing_result, healing_threshold, "Healed state"),
    ]

    for row_index, result, threshold, state_label in comparison_spec:
        opinions = result["agents"]["o_t"].to_numpy()
        ax_hist = axes[row_index, 0]
        ax_hist.hist(
            opinions,
            bins=28,
            color="#60a5fa" if row_index == 0 else "#34d399",
            edgecolor="black",
            alpha=0.85,
        )
        ax_hist.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax_hist.set_xlim(-1.05, 1.05)
        ax_hist.set_xlabel("Final opinion")
        ax_hist.set_ylabel("Agents")
        ax_hist.set_title(
            f"{state_label}: threshold={threshold:.1f}\n"
            f"variance={result['final_state']['final_opinion_variance']:.3f}, "
            f"extremists={result['final_state']['final_extremist_ratio']:.3f}"
        )
        ax_hist.grid(alpha=0.2)

        ax_network = axes[row_index, 1]
        graph = result["G_updated"]
        pos = result["pos"]
        node_colors = [graph.nodes[n]["opinion"] for n in graph.nodes()]
        nodes_draw = nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            cmap=plt.cm.coolwarm,
            node_size=36,
            ax=ax_network,
            vmin=-1,
            vmax=1,
        )
        nx.draw_networkx_edges(graph, pos, alpha=0.05, width=0.5, ax=ax_network)
        ax_network.set_title(
            f"{state_label}: homophily={result['final_state']['final_homophily_ratio']:.3f}, "
            f"cross-cutting={result['final_state']['final_cross_cutting_ratio']:.3f}"
        )
        ax_network.axis("off")
        fig.colorbar(nodes_draw, ax=ax_network, orientation="horizontal", shrink=0.72, pad=0.03)

    fig.tight_layout()
    return fig, axes


def load_leader_effects_outputs(output_dir: str | Path) -> dict[str, pd.DataFrame | dict]:
    base_dir = Path(output_dir)
    manifest_path = base_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "manifest": manifest,
        "experiment_grid_df": pd.read_csv(base_dir / "experiment_grid.csv"),
        "raw_df": pd.read_csv(base_dir / "raw" / "raw_results.csv"),
        "summary_df": pd.read_csv(base_dir / "summary" / "summary_results.csv"),
    }


def _mode_palette() -> dict[str, str]:
    return {
        "balanced": "#6b7280",
        "positive": "#dc2626",
        "negative": "#2563eb",
    }


def _share_palette(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    palette = sns.color_palette("crest", n_colors=len(labels))
    return {label: palette[index] for index, label in enumerate(labels)}


def _format_share_label(series: pd.Series) -> pd.Series:
    return series.apply(lambda value: f"{int(round(float(value) * 100))}%")


def plot_leader_effects_mode_comparison(summary_df: pd.DataFrame):
    n_values = sorted(summary_df["N"].unique())
    topology_values = sorted(summary_df["topology"].unique())
    share_order = [label for label in _format_share_label(pd.Series(sorted(summary_df["leader_share"].unique())))]

    fig, axes = plt.subplots(
        len(n_values),
        len(topology_values),
        figsize=(4.5 * len(topology_values), 3.6 * len(n_values)),
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    palette = _share_palette(share_order)

    for row_index, n_agents in enumerate(n_values):
        for col_index, topology in enumerate(topology_values):
            ax = axes[row_index, col_index]
            subset = summary_df[(summary_df["N"] == n_agents) & (summary_df["topology"] == topology)].copy()
            subset["leader_share_label"] = _format_share_label(subset["leader_share"])
            sns.barplot(
                data=subset,
                x="leader_mode",
                y="final_mean_opinion_mean",
                hue="leader_share_label",
                order=["balanced", "positive", "negative"],
                hue_order=share_order,
                palette=palette,
                ax=ax,
            )
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
            ax.set_title(f"N={n_agents} | {topology}")
            ax.set_xlabel("Leader mode")
            ax.set_ylabel("Final mean opinion")
            ax.grid(alpha=0.2, axis="y")
            if row_index != 0 or col_index != len(topology_values) - 1:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Leader share", loc="upper center", ncol=len(labels))
    fig.suptitle("Leader Mode Effects on Final Mean Opinion", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_leader_effects_content_balance(summary_df: pd.DataFrame):
    n_values = sorted(summary_df["N"].unique())
    topology_values = sorted(summary_df["topology"].unique())
    mode_order = ["balanced", "positive", "negative"]

    fig, axes = plt.subplots(
        len(n_values),
        len(topology_values),
        figsize=(4.5 * len(topology_values), 3.6 * len(n_values)),
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for row_index, n_agents in enumerate(n_values):
        for col_index, topology in enumerate(topology_values):
            ax = axes[row_index, col_index]
            subset = summary_df[(summary_df["N"] == n_agents) & (summary_df["topology"] == topology)].copy()
            subset["leader_share_label"] = _format_share_label(subset["leader_share"])
            sns.pointplot(
                data=subset,
                x="leader_share_label",
                y="content_balance_mean",
                hue="leader_mode",
                order=[f"{int(round(value * 100))}%" for value in sorted(summary_df["leader_share"].unique())],
                hue_order=mode_order,
                palette=_mode_palette(),
                dodge=0.25,
                markers=["o", "s", "^"],
                linestyles="-",
                errorbar=None,
                ax=ax,
            )
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
            ax.set_title(f"N={n_agents} | {topology}")
            ax.set_xlabel("Leader share")
            ax.set_ylabel("Support posts - oppose posts")
            ax.grid(alpha=0.2, axis="y")
            if row_index != 0 or col_index != len(topology_values) - 1:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Leader mode", loc="upper center", ncol=len(labels))
    fig.suptitle("Leader Share Effects on Net Content Supply", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_leader_effects_extremism(summary_df: pd.DataFrame):
    share_values = sorted(summary_df["leader_share"].unique())
    n_values = sorted(summary_df["N"].unique())

    fig, axes = plt.subplots(
        len(share_values),
        len(n_values),
        figsize=(4.4 * len(n_values), 3.4 * len(share_values)),
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for row_index, share in enumerate(share_values):
        for col_index, n_agents in enumerate(n_values):
            ax = axes[row_index, col_index]
            subset = summary_df[(summary_df["leader_share"] == share) & (summary_df["N"] == n_agents)].copy()
            sns.barplot(
                data=subset,
                x="topology",
                y="extremist_ratio_mean",
                hue="leader_mode",
                hue_order=["balanced", "positive", "negative"],
                palette=_mode_palette(),
                ax=ax,
            )
            ax.set_title(f"share={int(round(share * 100))}% | N={n_agents}")
            ax.set_xlabel("Topology")
            ax.set_ylabel("Extremist ratio")
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.2, axis="y")
            if row_index != 0 or col_index != len(n_values) - 1:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()

    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Leader mode", loc="upper center", ncol=len(labels))
    fig.suptitle("Topology Differences in Final Extremism", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_leader_effects_overview(summary_df: pd.DataFrame):
    benchmark_share = 0.03 if 0.03 in set(np.round(summary_df["leader_share"], 4)) else sorted(summary_df["leader_share"].unique())[0]
    subset = summary_df[np.isclose(summary_df["leader_share"], benchmark_share)].copy()

    metric_specs = [
        ("final_mean_opinion_mean", "Final mean opinion"),
        ("content_balance_mean", "Net content supply"),
        ("extremist_ratio_mean", "Extremist ratio"),
        ("homophily_ratio_mean", "Homophily ratio"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = np.atleast_2d(axes)

    for ax, (metric_column, ylabel) in zip(axes.ravel(), metric_specs):
        sns.lineplot(
            data=subset,
            x="N",
            y=metric_column,
            hue="leader_mode",
            hue_order=["balanced", "positive", "negative"],
            style="topology",
            palette=_mode_palette(),
            markers=True,
            dashes=False,
            ax=ax,
        )
        ax.set_xlabel("Population size N")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2, axis="y")

    axes[0, 0].set_title(f"Overview at leader share = {int(round(benchmark_share * 100))}%")
    for ax in axes.ravel()[1:]:
        ax.set_title("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.ravel():
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if handles:
        fig.legend(handles, labels, title="Leader mode", loc="upper center", ncol=len(labels))
    fig.tight_layout()
    return fig, axes


def plot_leader_effects_heatmap(summary_df: pd.DataFrame, metric: str = "final_mean_opinion_mean", fixed_n: int = 1000):
    subset = summary_df[summary_df["N"] == fixed_n].copy()
    if subset.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No summary rows available for N={fixed_n}", ha="center", va="center")
        ax.axis("off")
        return fig, np.asarray([ax])

    share_values = sorted(subset["leader_share"].unique())
    fig, axes = plt.subplots(1, len(share_values), figsize=(4.4 * len(share_values), 4.8), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, share in zip(axes, share_values):
        share_subset = subset[np.isclose(subset["leader_share"], share)].copy()
        share_subset = (
            share_subset.groupby(["topology", "leader_mode"], as_index=False)[metric]
            .mean()
        )
        pivot_df = share_subset.pivot(index="topology", columns="leader_mode", values=metric)
        pivot_df = pivot_df.reindex(index=sorted(pivot_df.index), columns=["balanced", "positive", "negative"])
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0.0,
            cbar=ax is axes[-1],
            ax=ax,
        )
        ax.set_title(f"N={fixed_n} | share={int(round(share * 100))}%")
        ax.set_xlabel("Leader mode")
        ax.set_ylabel("Topology")

    fig.suptitle(f"Heatmap of {metric.replace('_', ' ')}", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_no_leader_control_mean_opinion(summary_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=summary_df,
        x="N",
        y="final_mean_opinion_mean",
        hue="topology",
        style="topology",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_title("No-Leader Control: Final Mean Opinion")
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Final mean opinion")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig, ax


def plot_no_leader_control_content(summary_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=summary_df,
        x="N",
        y="content_balance_mean",
        hue="topology",
        style="topology",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_title("No-Leader Control: Net Content Supply")
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Support posts - oppose posts")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig, ax


def plot_no_leader_control_extremism(summary_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=summary_df,
        x="N",
        y="extremist_ratio_mean",
        hue="topology",
        style="topology",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title("No-Leader Control: Extremist Ratio")
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Extremist ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig, ax


def load_leader_effects_bundle(
    main_output_dir: str | Path,
    control_output_dir: str | Path,
) -> dict[str, dict[str, pd.DataFrame | dict]]:
    return {
        "main": load_leader_effects_outputs(main_output_dir),
        "control": load_leader_effects_outputs(control_output_dir),
    }


def summarize_leader_control_alignment(
    main_raw_df: pd.DataFrame,
    control_raw_df: pd.DataFrame,
) -> dict[str, object]:
    main_keys = (
        main_raw_df[["N", "topology", "T_rounds"]]
        .drop_duplicates()
        .sort_values(["N", "topology", "T_rounds"])
    )
    control_keys = (
        control_raw_df[["N", "topology", "T_rounds"]]
        .drop_duplicates()
        .sort_values(["N", "topology", "T_rounds"])
    )

    main_seed_count = int(main_raw_df["seed"].nunique()) if not main_raw_df.empty else 0
    control_seed_count = int(control_raw_df["seed"].nunique()) if not control_raw_df.empty else 0

    return {
        "main_condition_count": int(len(main_raw_df)),
        "control_condition_count": int(len(control_raw_df)),
        "main_seed_count": main_seed_count,
        "control_seed_count": control_seed_count,
        "shared_population_sizes": sorted(set(main_raw_df["N"]).intersection(set(control_raw_df["N"]))),
        "shared_topologies": sorted(set(main_raw_df["topology"]).intersection(set(control_raw_df["topology"]))),
        "shared_rounds": sorted(set(main_raw_df["T_rounds"]).intersection(set(control_raw_df["T_rounds"]))),
        "grid_alignment_ok": main_keys.reset_index(drop=True).equals(control_keys.reset_index(drop=True)),
    }


def build_leader_control_comparison_summary(
    main_summary_df: pd.DataFrame,
    control_summary_df: pd.DataFrame,
    benchmark_share: float = 0.03,
) -> pd.DataFrame:
    benchmark_candidates = sorted(main_summary_df["leader_share"].unique())
    if benchmark_share not in set(np.round(benchmark_candidates, 4)):
        benchmark_share = benchmark_candidates[0]

    main_subset = main_summary_df[np.isclose(main_summary_df["leader_share"], benchmark_share)].copy()
    main_subset["comparison_group"] = main_subset["leader_mode"].map(
        {
            "balanced": f"balanced ({int(round(benchmark_share * 100))}%)",
            "positive": f"positive ({int(round(benchmark_share * 100))}%)",
            "negative": f"negative ({int(round(benchmark_share * 100))}%)",
        }
    )
    main_subset["comparison_order"] = main_subset["leader_mode"].map(
        {"balanced": 1, "positive": 2, "negative": 3}
    )

    control_subset = control_summary_df.copy()
    control_subset["comparison_group"] = "no leader"
    control_subset["comparison_order"] = 0

    combined = pd.concat([control_subset, main_subset], ignore_index=True, sort=False)
    combined["benchmark_share"] = float(benchmark_share)
    return combined.sort_values(["N", "topology", "comparison_order"]).reset_index(drop=True)


def build_leader_control_comparison_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "N",
        "topology",
        "comparison_group",
        "final_mean_opinion_mean",
        "final_mean_opinion_ci_low",
        "final_mean_opinion_ci_high",
        "content_balance_mean",
        "content_balance_ci_low",
        "content_balance_ci_high",
        "extremist_ratio_mean",
        "extremist_ratio_ci_low",
        "extremist_ratio_ci_high",
        "homophily_ratio_mean",
    ]
    return comparison_df[columns].copy()


def _comparison_palette(groups: list[str]) -> dict[str, str]:
    palette = {
        "no leader": "#111827",
    }
    for group in groups:
        if group.startswith("balanced"):
            palette[group] = "#6b7280"
        elif group.startswith("positive"):
            palette[group] = "#dc2626"
        elif group.startswith("negative"):
            palette[group] = "#2563eb"
    return palette


def plot_leader_control_comparison(
    comparison_df: pd.DataFrame,
    metric_column: str,
    ylabel: str,
    title: str,
):
    n_values = sorted(comparison_df["N"].unique())
    topology_values = sorted(comparison_df["topology"].unique())
    group_order = (
        comparison_df[["comparison_group", "comparison_order"]]
        .drop_duplicates()
        .sort_values("comparison_order")["comparison_group"]
        .tolist()
    )

    fig, axes = plt.subplots(
        len(n_values),
        len(topology_values),
        figsize=(4.8 * len(topology_values), 3.8 * len(n_values)),
        sharey=True,
    )
    axes = np.atleast_2d(axes)
    palette = _comparison_palette(group_order)

    for row_index, n_agents in enumerate(n_values):
        for col_index, topology in enumerate(topology_values):
            ax = axes[row_index, col_index]
            subset = comparison_df[
                (comparison_df["N"] == n_agents) & (comparison_df["topology"] == topology)
            ].copy()
            sns.barplot(
                data=subset,
                x="comparison_group",
                y=metric_column,
                hue="comparison_group",
                order=group_order,
                hue_order=group_order,
                palette=palette,
                dodge=False,
                legend=False,
                ax=ax,
            )
            ax.set_title(f"N={n_agents} | {topology}")
            ax.set_xlabel("")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2, axis="y")
            ax.tick_params(axis="x", rotation=20)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_leader_control_mean_opinion(comparison_df: pd.DataFrame):
    return plot_leader_control_comparison(
        comparison_df,
        metric_column="final_mean_opinion_mean",
        ylabel="Final mean opinion",
        title="No-Leader Control vs 3% Leader Benchmark",
    )


def plot_leader_control_content_balance(comparison_df: pd.DataFrame):
    return plot_leader_control_comparison(
        comparison_df,
        metric_column="content_balance_mean",
        ylabel="Support posts - oppose posts",
        title="No-Leader Control vs 3% Leader Benchmark: Net Content Supply",
    )


def plot_leader_control_extremism(comparison_df: pd.DataFrame):
    fig, axes = plot_leader_control_comparison(
        comparison_df,
        metric_column="extremist_ratio_mean",
        ylabel="Extremist ratio",
        title="No-Leader Control vs 3% Leader Benchmark: Extremism",
    )
    for ax in np.atleast_1d(axes).ravel():
        ax.set_ylim(0.0, 1.0)
    return fig, axes


def _thesis_theme():
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
        },
    )


def _thesis_mode_palette() -> dict[str, str]:
    return {
        "balanced": "#6b7280",
        "positive": "#b91c1c",
        "negative": "#1d4ed8",
    }


def _group_mean_sem(df: pd.DataFrame, value_columns: list[str], group_columns: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_columns, as_index=False)[value_columns].agg(["mean", "std", "count"])
    grouped.columns = [
        "_".join([part for part in column if part]).strip("_")
        for column in grouped.columns.to_flat_index()
    ]
    for column in value_columns:
        grouped[f"{column}_sem"] = grouped[f"{column}_std"] / np.sqrt(grouped[f"{column}_count"].clip(lower=1))
        grouped[f"{column}_ci_low"] = grouped[f"{column}_mean"] - 1.96 * grouped[f"{column}_sem"].fillna(0.0)
        grouped[f"{column}_ci_high"] = grouped[f"{column}_mean"] + 1.96 * grouped[f"{column}_sem"].fillna(0.0)
    return grouped


def build_no_leader_reference_summary(control_summary_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "final_mean_opinion_mean",
        "content_balance_mean",
        "extremist_ratio_mean",
    ]
    return _group_mean_sem(control_summary_df, columns, ["N", "topology"])


def build_rq2_share_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "content_balance_mean",
        "extremist_ratio_mean",
        "homophily_ratio_mean",
    ]
    grouped = _group_mean_sem(summary_df, columns, ["leader_share", "leader_mode"])
    grouped["leader_share_pct"] = grouped["leader_share"].astype(float) * 100.0
    return grouped.sort_values(["leader_share_pct", "leader_mode"]).reset_index(drop=True)


def build_rq3_delta_summary(
    summary_df: pd.DataFrame,
    control_summary_df: pd.DataFrame,
    benchmark_share: float = 0.03,
) -> pd.DataFrame:
    subset = summary_df[np.isclose(summary_df["leader_share"], benchmark_share)].copy()
    control = control_summary_df[["N", "topology", "final_mean_opinion_mean", "extremist_ratio_mean", "homophily_ratio_mean"]].copy()
    control = control.rename(
        columns={
            "final_mean_opinion_mean": "control_final_mean_opinion_mean",
            "extremist_ratio_mean": "control_extremist_ratio_mean",
            "homophily_ratio_mean": "control_homophily_ratio_mean",
        }
    )
    merged = subset.merge(control, on=["N", "topology"], how="left")
    merged["delta_final_mean_opinion"] = (
        merged["final_mean_opinion_mean"] - merged["control_final_mean_opinion_mean"]
    )
    merged["delta_extremist_ratio"] = (
        merged["extremist_ratio_mean"] - merged["control_extremist_ratio_mean"]
    )
    merged["delta_homophily_ratio"] = (
        merged["homophily_ratio_mean"] - merged["control_homophily_ratio_mean"]
    )
    return merged


def plot_thesis_no_leader_benchmark(control_summary_df: pd.DataFrame):
    _thesis_theme()
    grouped = build_no_leader_reference_summary(control_summary_df)
    topology_order = sorted(grouped["topology"].unique())
    palette = sns.color_palette("crest", n_colors=len(topology_order))
    color_map = {topology: palette[index] for index, topology in enumerate(topology_order)}

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharex=True)
    metric_specs = [
        ("final_mean_opinion_mean", "Directional Drift", "Final mean opinion"),
        ("content_balance_mean", "Net Content Supply", "Support posts - oppose posts"),
        ("extremist_ratio_mean", "Opinion Extremity", "Extremist ratio"),
    ]

    for ax, (metric, title, ylabel) in zip(axes, metric_specs):
        for topology in topology_order:
            subset = grouped[grouped["topology"] == topology].sort_values("N")
            ax.plot(
                subset["N"],
                subset[f"{metric}_mean"],
                marker="o",
                linewidth=2.0,
                color=color_map[topology],
                label=topology,
            )
            ax.fill_between(
                subset["N"],
                subset[f"{metric}_ci_low"],
                subset[f"{metric}_ci_high"],
                color=color_map[topology],
                alpha=0.12,
            )
        if metric == "final_mean_opinion_mean":
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        if metric == "extremist_ratio_mean":
            ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.set_xlabel("Population size N")
        ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Topology", loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig, axes


def plot_thesis_rq1_orientation(
    summary_df: pd.DataFrame,
    control_summary_df: pd.DataFrame,
    benchmark_share: float = 0.03,
):
    _thesis_theme()
    subset = summary_df[np.isclose(summary_df["leader_share"], benchmark_share)].copy()
    control = control_summary_df.copy()
    topology_order = sorted(subset["topology"].unique())
    mode_order = ["balanced", "positive", "negative"]
    mode_palette = _thesis_mode_palette()

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for ax, topology in zip(axes.ravel(), topology_order):
        topo_control = control[control["topology"] == topology].sort_values("N")
        ax.plot(
            topo_control["N"],
            topo_control["final_mean_opinion_mean"],
            color="#111827",
            linewidth=1.8,
            linestyle="--",
            marker="o",
            label="no leader",
        )

        topo_subset = subset[subset["topology"] == topology].copy()
        for leader_mode in mode_order:
            mode_subset = topo_subset[topo_subset["leader_mode"] == leader_mode].sort_values("N")
            ax.plot(
                mode_subset["N"],
                mode_subset["final_mean_opinion_mean"],
                color=mode_palette[leader_mode],
                linewidth=2.2,
                marker="o",
                label=leader_mode,
            )
        ax.axhline(0.0, color="black", linewidth=0.9, linestyle=":", alpha=0.5)
        ax.set_title(topology)
        ax.set_xlabel("Population size N")
        ax.set_ylabel("Final mean opinion")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Condition", loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig, axes


def plot_thesis_rq2_share_gradients(summary_df: pd.DataFrame):
    _thesis_theme()
    grouped = build_rq2_share_summary(summary_df)
    mode_order = ["balanced", "positive", "negative"]
    mode_palette = _thesis_mode_palette()

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.4), sharex=True)
    metric_specs = [
        ("content_balance_mean", "Net Content Supply", "Support posts - oppose posts"),
        ("extremist_ratio_mean", "Opinion Extremity", "Extremist ratio"),
        ("homophily_ratio_mean", "Network Segregation", "Homophily ratio"),
    ]

    for ax, (metric, title, ylabel) in zip(axes, metric_specs):
        for leader_mode in mode_order:
            mode_subset = grouped[grouped["leader_mode"] == leader_mode].sort_values("leader_share_pct")
            ax.plot(
                mode_subset["leader_share_pct"],
                mode_subset[f"{metric}_mean"],
                color=mode_palette[leader_mode],
                linewidth=2.2,
                marker="o",
                label=leader_mode,
            )
            ax.fill_between(
                mode_subset["leader_share_pct"],
                mode_subset[f"{metric}_ci_low"],
                mode_subset[f"{metric}_ci_high"],
                color=mode_palette[leader_mode],
                alpha=0.12,
            )
        if metric != "content_balance_mean":
            ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.set_xlabel("Leader share (%)")
        ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Leader orientation", loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig, axes


def plot_thesis_rq3_robustness(
    summary_df: pd.DataFrame,
    control_summary_df: pd.DataFrame,
    benchmark_share: float = 0.03,
):
    _thesis_theme()
    deltas = build_rq3_delta_summary(
        summary_df,
        control_summary_df,
        benchmark_share=benchmark_share,
    )
    mode_palette = _thesis_mode_palette()
    topology_styles = {topology: style for topology, style in zip(sorted(deltas["topology"].unique()), ["-", "--", "-.", ":"])}

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharex=True)
    metric_specs = [
        ("delta_final_mean_opinion", "Directional Effect Relative to Control", r"$\Delta$ final mean opinion"),
        ("delta_extremist_ratio", "Extremity Effect Relative to Control", r"$\Delta$ extremist ratio"),
    ]

    for ax, (metric, title, ylabel) in zip(axes, metric_specs):
        for topology in sorted(deltas["topology"].unique()):
            topo_subset = deltas[deltas["topology"] == topology].copy()
            for leader_mode in ["balanced", "positive", "negative"]:
                mode_subset = topo_subset[topo_subset["leader_mode"] == leader_mode].sort_values("N")
                ax.plot(
                    mode_subset["N"],
                    mode_subset[metric],
                    color=mode_palette[leader_mode],
                    linestyle=topology_styles[topology],
                    linewidth=1.9,
                    marker="o",
                )
        ax.axhline(0.0, color="black", linewidth=0.9, linestyle=":", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Population size N")
        ax.set_ylabel(ylabel)

    mode_handles = [
        plt.Line2D([0], [0], color=mode_palette[mode], linewidth=2.2, marker="o", label=mode)
        for mode in ["balanced", "positive", "negative"]
    ]
    topo_handles = [
        plt.Line2D([0], [0], color="#374151", linewidth=1.9, linestyle=topology_styles[topology], label=topology)
        for topology in sorted(deltas["topology"].unique())
    ]
    fig.legend(mode_handles + topo_handles, [handle.get_label() for handle in mode_handles + topo_handles], title="Leader orientation / topology", loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig, axes


def plot_thesis_ch4_synthesis(summary_df: pd.DataFrame, benchmark_share: float = 0.03):
    _thesis_theme()
    subset = summary_df[np.isclose(summary_df["leader_share"], benchmark_share)].copy()
    mode_palette = _thesis_mode_palette()
    metric_specs = [
        ("final_mean_opinion_mean", "Final mean opinion"),
        ("content_balance_mean", "Support posts - oppose posts"),
        ("extremist_ratio_mean", "Extremist ratio"),
        ("homophily_ratio_mean", "Homophily ratio"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex=True)
    axes = np.atleast_2d(axes)

    for ax, (metric, ylabel) in zip(axes.ravel(), metric_specs):
        sns.lineplot(
            data=subset,
            x="N",
            y=metric,
            hue="leader_mode",
            hue_order=["balanced", "positive", "negative"],
            style="topology",
            palette=mode_palette,
            markers=True,
            dashes=True,
            linewidth=2.0,
            ax=ax,
        )
        if metric == "final_mean_opinion_mean":
            ax.axhline(0.0, color="black", linewidth=0.9, linestyle=":", alpha=0.5)
        if metric in {"extremist_ratio_mean", "homophily_ratio_mean"}:
            ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Population size N")
        ax.set_ylabel(ylabel)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Leader orientation / topology", loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig, axes


def plot_thesis_baseline_summary(
    history_df: pd.DataFrame,
    agents_df: pd.DataFrame,
    params: dict | None = None,
):
    _thesis_theme()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    ax_traj = axes[0]
    ax_traj.plot(
        history_df["round"],
        history_df["mean_opinion"],
        color="#1d4ed8",
        linewidth=2.4,
        marker="o",
        markersize=3.5,
    )
    ax_traj.axhline(0.0, color="black", linewidth=0.9, linestyle=":", alpha=0.5)
    ax_traj.set_title("Directional Drift")
    ax_traj.set_xlabel("Round")
    ax_traj.set_ylabel("Mean opinion")
    ax_traj.set_ylim(-1.0, 0.05)

    ax_pol = axes[1]
    ax_pol.plot(
        history_df["round"],
        history_df["extremist_ratio"],
        color="#b91c1c",
        linewidth=2.2,
        label="Extremist ratio",
    )
    ax_pol.plot(
        history_df["round"],
        history_df["homophily_ratio"],
        color="#374151",
        linewidth=2.0,
        linestyle="--",
        label="Homophily ratio",
    )
    ax_pol.set_title("Polarization and Segregation")
    ax_pol.set_xlabel("Round")
    ax_pol.set_ylabel("Ratio")
    ax_pol.set_ylim(0.0, 1.0)
    ax_pol.legend(frameon=True, loc="lower right")

    ax_dist = axes[2]
    counts, bin_edges, _ = ax_dist.hist(
        agents_df["o_t"],
        bins=24,
        color="#93c5fd",
        edgecolor="#1f2937",
        alpha=0.95,
    )
    if params is not None:
        initial_mean = float(params.get("opinion_mean", 0.0))
        initial_std = float(params.get("opinion_std", 0.0))
        if initial_std > 0:
            x_values = np.linspace(-1.0, 1.0, 400)
            bin_width = bin_edges[1] - bin_edges[0]
            density = (
                np.exp(-0.5 * ((x_values - initial_mean) / initial_std) ** 2)
                / (initial_std * np.sqrt(2.0 * np.pi))
            )
            scaled_density = len(agents_df) * bin_width * density
            ax_dist.plot(
                x_values,
                scaled_density,
                color="#dc2626",
                linewidth=2.1,
                linestyle="--",
                label="Initial distribution",
            )
            ax_dist.legend(frameon=True, loc="upper right")
    ax_dist.set_title("Final Opinion Distribution")
    ax_dist.set_xlabel("Latent opinion")
    ax_dist.set_ylabel("Agents")
    ax_dist.set_xlim(-1.05, 1.05)

    fig.tight_layout()
    return fig, axes
