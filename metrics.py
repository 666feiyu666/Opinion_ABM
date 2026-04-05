from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_history_frame(round_records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(round_records)


def build_opinion_trajectory_frame(opinion_snapshots: list[dict]) -> pd.DataFrame:
    if not opinion_snapshots:
        return pd.DataFrame(columns=["round", "node", "opinion"])
    return pd.DataFrame(opinion_snapshots)


def format_round_summary(round_number: int, summary: dict) -> str:
    return (
        f"Round {round_number:02d} | "
        f"Creators={int(summary['actual_creators']):3d} | "
        f"Support={int(summary['support_posts']):3d} | "
        f"Oppose={int(summary['oppose_posts']):3d} | "
        f"Toxic={int(summary['toxic_posts']):3d} | "
        f"AvgExposure={summary['avg_exposure_size']:.2f} | "
        f"Edges={int(summary['edge_count'])}"
    )


def compute_average_neighbor_opinions(graph) -> list[float]:
    averages = []
    for node in graph.nodes():
        neighbors = list(graph.successors(node))
        if neighbors:
            avg_opinion = sum(graph.nodes[nb]["opinion"] for nb in neighbors) / len(neighbors)
        else:
            avg_opinion = graph.nodes[node]["opinion"]
        averages.append(avg_opinion)
    return averages


def compute_opinion_variance(agents) -> float:
    return float(np.var(agents["o_t"].to_numpy()))


def compute_extremist_ratio(agents, opinion_threshold: float = 0.7) -> float:
    opinions = agents["o_t"].to_numpy()
    return float(np.mean(np.abs(opinions) > opinion_threshold))


def compute_homophily_ratio(graph, neutral_band: float = 0.0) -> float:
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return float("nan")

    same_side_edges = 0
    comparable_edges = 0

    for source, target in graph.edges():
        source_opinion = graph.nodes[source]["opinion"]
        target_opinion = graph.nodes[target]["opinion"]

        if abs(source_opinion) <= neutral_band or abs(target_opinion) <= neutral_band:
            continue

        comparable_edges += 1
        if np.sign(source_opinion) == np.sign(target_opinion):
            same_side_edges += 1

    if comparable_edges == 0:
        return float("nan")
    return float(same_side_edges / comparable_edges)


def compute_sign_modularity(graph, neutral_band: float = 0.0) -> float:
    graph_undirected = graph.to_undirected()
    if graph_undirected.number_of_edges() == 0:
        return float("nan")

    negative_nodes = set()
    neutral_nodes = set()
    positive_nodes = set()

    for node, attrs in graph_undirected.nodes(data=True):
        opinion = attrs.get("opinion", 0.0)
        if opinion < -neutral_band:
            negative_nodes.add(node)
        elif opinion > neutral_band:
            positive_nodes.add(node)
        else:
            neutral_nodes.add(node)

    communities = [community for community in (negative_nodes, neutral_nodes, positive_nodes) if community]
    if len(communities) <= 1:
        return 0.0

    return float(nx.algorithms.community.quality.modularity(graph_undirected, communities))


def summarize_final_state(
    graph,
    agents,
    extremist_threshold: float = 0.7,
    neutral_band: float = 0.0,
) -> dict:
    homophily_ratio = compute_homophily_ratio(
        graph,
        neutral_band=neutral_band,
    )
    return {
        "final_mean_opinion": float(agents["o_t"].mean()),
        "final_std_opinion": float(agents["o_t"].std()),
        "final_opinion_variance": compute_opinion_variance(agents),
        "final_extremist_ratio": compute_extremist_ratio(
            agents,
            opinion_threshold=extremist_threshold,
        ),
        "final_homophily_ratio": homophily_ratio,
        "final_cross_cutting_ratio": float(1.0 - homophily_ratio),
        "final_sign_modularity": compute_sign_modularity(
            graph,
            neutral_band=neutral_band,
        ),
        "final_edge_count": int(graph.number_of_edges()),
    }


def summarize_metric_distribution(values, confidence: float = 0.95) -> dict:
    array = np.asarray(values, dtype=float)
    clean = array[~np.isnan(array)]
    if clean.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
        }

    mean_value = float(clean.mean())
    std_value = float(clean.std(ddof=1)) if clean.size > 1 else 0.0
    sem_value = float(std_value / np.sqrt(clean.size)) if clean.size > 1 else 0.0
    z_value = 1.96 if confidence == 0.95 else 1.96
    ci_half_width = z_value * sem_value
    return {
        "mean": mean_value,
        "std": std_value,
        "sem": sem_value,
        "ci_low": float(mean_value - ci_half_width),
        "ci_high": float(mean_value + ci_half_width),
        "n": int(clean.size),
    }
