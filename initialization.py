from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd

from utils import sample_stance_from_opinion, sign_opinion, tanh_mapping


def _normalize_topology_name(params: dict) -> str:
    return str(params.get("network_topology", "BA")).strip().upper()


def _target_average_degree(params: dict) -> float:
    explicit_target = params.get("target_average_degree")
    if explicit_target is not None:
        return float(explicit_target)
    return float(2 * int(params["m_BA"]))


def _balanced_block_sizes(n_agents: int, num_blocks: int) -> list[int]:
    base_size = n_agents // num_blocks
    remainder = n_agents % num_blocks
    return [base_size + (1 if block_index < remainder else 0) for block_index in range(num_blocks)]


def _build_undirected_graph(params: dict, seed: int):
    n_agents = int(params["N"])
    topology = _normalize_topology_name(params)
    target_average_degree = _target_average_degree(params)

    if topology == "BA":
        m_ba = int(params["m_BA"])
        graph = nx.barabasi_albert_graph(n=n_agents, m=m_ba, seed=seed)
        metadata = {
            "network_topology": "BA",
            "target_average_degree": float(2 * m_ba),
        }
        return graph, metadata

    if topology == "ER":
        edge_probability = params.get("er_p")
        if edge_probability is None:
            edge_probability = min(1.0, target_average_degree / max(n_agents - 1, 1))
        graph = nx.erdos_renyi_graph(n=n_agents, p=float(edge_probability), seed=seed)
        metadata = {
            "network_topology": "ER",
            "er_p": float(edge_probability),
            "target_average_degree": float(target_average_degree),
        }
        return graph, metadata

    if topology == "WS":
        ws_k = int(params.get("ws_k", max(2, round(target_average_degree))))
        if ws_k % 2 != 0:
            ws_k += 1
        ws_k = min(ws_k, n_agents - 1 if (n_agents - 1) % 2 == 0 else n_agents - 2)
        ws_k = max(ws_k, 2)
        rewire_probability = float(params.get("ws_rewire_prob", 0.10))
        graph = nx.watts_strogatz_graph(
            n=n_agents,
            k=ws_k,
            p=rewire_probability,
            seed=seed,
        )
        metadata = {
            "network_topology": "WS",
            "ws_k": int(ws_k),
            "ws_rewire_prob": float(rewire_probability),
            "target_average_degree": float(target_average_degree),
        }
        return graph, metadata

    if topology == "SBM":
        num_blocks = max(2, int(params.get("sbm_num_blocks", 4)))
        within_share = float(params.get("sbm_within_share", 0.70))
        within_share = float(np.clip(within_share, 0.05, 0.95))
        block_sizes = _balanced_block_sizes(n_agents, num_blocks)

        within_targets = [max(size - 1, 1) for size in block_sizes]
        outside_targets = [max(n_agents - size, 1) for size in block_sizes]

        p_in_values = [
            min(1.0, (target_average_degree * within_share) / target)
            for target in within_targets
        ]
        p_out_values = [
            min(1.0, (target_average_degree * (1.0 - within_share)) / target)
            for target in outside_targets
        ]

        probability_matrix = []
        for row_index in range(num_blocks):
            row_values = []
            for col_index in range(num_blocks):
                if row_index == col_index:
                    row_values.append(float(p_in_values[row_index]))
                else:
                    row_values.append(float(min(p_out_values[row_index], p_out_values[col_index])))
            probability_matrix.append(row_values)

        graph = nx.stochastic_block_model(
            sizes=block_sizes,
            p=probability_matrix,
            seed=seed,
        )
        metadata = {
            "network_topology": "SBM",
            "sbm_num_blocks": int(num_blocks),
            "sbm_within_share": float(within_share),
            "sbm_block_sizes": list(block_sizes),
            "sbm_p_in_mean": float(np.mean(p_in_values)),
            "sbm_p_out_mean": float(np.mean(p_out_values)),
            "target_average_degree": float(target_average_degree),
        }
        return graph, metadata

    raise ValueError(f"Unsupported network_topology: {topology}")


def _orient_graph(undirected_graph, rng_local: np.random.Generator) -> nx.DiGraph:
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(undirected_graph.nodes())

    for source, target in undirected_graph.edges():
        if rng_local.random() < 0.5:
            directed_graph.add_edge(source, target)
        else:
            directed_graph.add_edge(target, source)

    return directed_graph


def _select_leaders(agents: pd.DataFrame, params: dict, rng_local: np.random.Generator) -> pd.DataFrame:
    agents = agents.copy()
    agents["L"] = 0

    leader_share = params.get("leader_share")
    selection_method = str(params.get("leader_selection_method", "threshold")).strip().lower()

    if leader_share is None:
        threshold = params["leader_in_degree_threshold"]
        agents["L"] = (agents["F_t"] > threshold).astype(int)
        return agents

    if selection_method == "none" or float(leader_share) <= 0.0:
        return agents

    leader_count = max(1, int(round(float(leader_share) * len(agents))))
    leader_count = min(leader_count, len(agents))

    if selection_method == "random":
        chosen_nodes = rng_local.choice(agents.index.to_numpy(), size=leader_count, replace=False)
    else:
        ranked_agents = agents.sort_values(["F_t", "node"], ascending=[False, True])
        chosen_nodes = ranked_agents.head(leader_count).index.to_numpy()

    agents.loc[chosen_nodes, "L"] = 1
    return agents


def _leader_signs(rng_local: np.random.Generator, count: int, mode: str) -> np.ndarray:
    if mode == "positive":
        return np.ones(count)
    if mode == "negative":
        return -np.ones(count)
    return rng_local.choice([-1.0, 1.0], size=count)


def initialize_model(params: dict, seed: int = 42):
    rng_local = np.random.default_rng(seed)
    np.random.seed(seed)

    n_agents = int(params["N"])
    graph_undirected, topology_metadata = _build_undirected_graph(params, seed=seed)
    graph_directed = _orient_graph(graph_undirected, rng_local)

    agents = pd.DataFrame(index=range(n_agents))
    agents["node"] = agents.index
    agents["F_t"] = [graph_directed.in_degree(i) for i in range(n_agents)]
    agents = _select_leaders(agents, params, rng_local)

    opinions = rng_local.normal(
        loc=params["opinion_mean"],
        scale=params["opinion_std"],
        size=n_agents,
    )
    agents["o_t"] = np.clip(opinions, -1.0, 1.0)
    agents["tau_t"] = rng_local.normal(
        loc=params["tau_init_mean"],
        scale=params["tau_init_std"],
        size=n_agents,
    )
    agents["e_t"] = rng_local.normal(
        loc=params.get("e_init_mean", 0.12),
        scale=params.get("e_init_std", 0.05),
        size=n_agents,
    )

    is_leader = agents["L"] == 1
    agents.loc[is_leader, "tau_t"] = rng_local.normal(
        loc=params["tau_L_init_mean"],
        scale=params["tau_L_init_std"],
        size=int(is_leader.sum()),
    )
    agents.loc[is_leader, "e_t"] = rng_local.normal(
        loc=params.get("e_L_init_mean", 0.90),
        scale=params.get("e_L_init_std", 0.05),
        size=int(is_leader.sum()),
    )

    leader_count = int(is_leader.sum())
    leader_signs = _leader_signs(
        rng_local,
        count=leader_count,
        mode=str(params.get("leader_opinion_mode", "balanced")).lower(),
    )
    leader_magnitudes = rng_local.uniform(0.6, 0.9, size=leader_count)
    agents.loc[is_leader, "o_t"] = leader_signs * leader_magnitudes

    agents["tau_t"] = np.clip(agents["tau_t"], 0.1, params["tau_max"])
    agents["tau_t1"] = agents["tau_t"].copy()
    agents["e_t"] = np.clip(agents["e_t"], 0.0, params.get("involvement_max", 1.0))
    agents["e_t1"] = agents["e_t"].copy()
    agents["confidence"] = agents["tau_t"].copy()
    agents["s_t"] = agents.apply(
        lambda row: sample_stance_from_opinion(
            opinion=row["o_t"],
            confidence=row["tau_t"],
            eta_expression=params.get("eta_expression", 1.0),
            rng=rng_local,
        ),
        axis=1,
    )
    agents.loc[is_leader, "s_t"] = agents.loc[is_leader, "o_t"].apply(sign_opinion)

    agents["Abar"] = rng_local.uniform(
        params["Abar_low"],
        params["Abar_high"],
        size=n_agents,
    )
    agents["A_t"] = agents["Abar"].copy()
    agents["m_t"] = agents["o_t"].apply(lambda x: tanh_mapping(x, params["kappa"]))

    agents["O_t"] = 0
    agents["C_t"] = 0

    agents["M_pC_prev"] = 0.0
    agents["M_pT_prev"] = 0.0
    agents["M_nC_prev"] = 0.0
    agents["M_nT_prev"] = 0.0

    blocks = {i: set() for i in range(n_agents)}
    pos = nx.spring_layout(graph_undirected, seed=seed, k=min(0.25, 5.0 / math.sqrt(max(n_agents, 1))), iterations=100)

    for node in graph_directed.nodes():
        graph_directed.nodes[node]["opinion"] = agents.at[node, "o_t"]
        graph_directed.nodes[node]["leader"] = agents.at[node, "L"]

    initialization_metadata = {
        **topology_metadata,
        "leader_count": int(leader_count),
        "realized_leader_share": float(leader_count / max(n_agents, 1)),
        "leader_selection_method": str(params.get("leader_selection_method", "threshold")),
    }

    return graph_directed, graph_undirected, agents, blocks, pos, initialization_metadata
