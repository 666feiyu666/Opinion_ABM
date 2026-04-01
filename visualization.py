from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

from metrics import compute_average_neighbor_opinions


def plot_time_series_summaries(history_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    series_spec = [
        ("actual_creators", "Number of Content Creators by Round", "Creators"),
        ("avg_exposure_size", "Average Exposure Size by Round", "Average Exposure Size"),
        ("mean_opinion", "Mean Latent Opinion by Round", "Mean Opinion"),
        ("edge_count", "Directed Edge Count by Round", "Number of Edges"),
    ]

    for ax, (column, title, ylabel) in zip(axes, series_spec):
        ax.plot(history_df["round"], history_df[column], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

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


def plot_final_opinion_distribution(agents):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(agents["o_t"], bins=25, edgecolor="black")
    ax.set_title("Final Latent Opinion Distribution After Simulation")
    ax.set_xlabel("Latent Opinion")
    ax.set_ylabel("Number of Agents")
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
