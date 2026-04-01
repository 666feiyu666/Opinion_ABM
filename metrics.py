from __future__ import annotations

import pandas as pd


def build_history_frame(round_records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(round_records)


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
