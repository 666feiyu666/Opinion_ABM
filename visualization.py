from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opinion_abm_mpl")

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
