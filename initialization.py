from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from utils import sign_opinion, tanh_mapping


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
    agents["s_t"] = agents["o_t"].apply(sign_opinion)

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

    agents["confidence"] = 1.0

    blocks = {i: set() for i in range(n_agents)}
    pos = nx.spring_layout(g_undirected, seed=seed, k=0.25, iterations=100)

    return g_directed, g_undirected, agents, blocks, pos
