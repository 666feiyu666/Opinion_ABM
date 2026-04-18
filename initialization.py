from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from utils import sample_stance_from_opinion, sign_opinion, tanh_mapping


def _leader_signs(rng_local: np.random.Generator, count: int, mode: str) -> np.ndarray:
    if mode == "positive":
        return np.ones(count)
    if mode == "negative":
        return -np.ones(count)
    return rng_local.choice([-1.0, 1.0], size=count)


def initialize_model(params: dict, seed: int = 42):
    rng_local = np.random.default_rng(seed)
    np.random.seed(seed)

    n_agents = params["N"]
    m_ba = params["m_BA"]

    g_undirected = nx.barabasi_albert_graph(n=n_agents, m=m_ba, seed=seed)
    g_directed = nx.DiGraph()
    g_directed.add_nodes_from(g_undirected.nodes())

    for u, v in g_undirected.edges():
        if rng_local.random() < 0.5:
            g_directed.add_edge(u, v)
        else:
            g_directed.add_edge(v, u)

    agents = pd.DataFrame(index=range(n_agents))
    agents["node"] = agents.index
    agents["F_t"] = [g_directed.in_degree(i) for i in range(n_agents)]

    threshold = params["leader_in_degree_threshold"]
    agents["L"] = (agents["F_t"] > threshold).astype(int)

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

    # 两极化领袖：将领袖的潜在意见强制推向光谱两端，作为极化扩散的“引擎”。
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
    pos = nx.spring_layout(g_undirected, seed=seed, k=0.25, iterations=100)

    return g_directed, g_undirected, agents, blocks, pos
