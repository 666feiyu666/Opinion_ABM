"""Matched undirected topology parameters before independent edge orientation.

Historical source: legacy c33a230, initialization.py (_build_undirected_graph)
and config.py. Its experiments/base.py retained initial edge counts and generator
metadata; no dedicated generator invariant test suite was found there.
We retain BA m=3, WS k=6/p=.10, and four balanced SBM blocks with within share .70.
The OLIM 2.0 design instead targets m*(N-m) edges for ER/SBM (historical target:
mean undirected degree 2*m). SBM probabilities divide expected within/between
edge counts by the corresponding numbers of possible unordered pairs. Invalid
parameters fail rather than adopting the legacy clipping rules. Edge orientation
and separate belief/leader random streams remain those of the current model.
These defaults implement provisional design settings, not calibrated estimates.
"""

from dataclasses import dataclass, asdict
from math import isfinite

import networkx as nx


@dataclass(frozen=True)
class TopologyConfig:
    family: str = "ba"
    ws_k: int = 6
    ws_rewire_probability: float = 0.10
    sbm_blocks: int = 4
    sbm_within_share: float = 0.70

    def __post_init__(self):
        if self.family not in ("ba", "er", "ws", "sbm"):
            raise ValueError("Unknown topology family")
        if type(self.ws_k) is not int or self.ws_k < 2 or self.ws_k % 2:
            raise ValueError("ws_k must be a positive even integer")
        if type(self.sbm_blocks) is not int or self.sbm_blocks < 2:
            raise ValueError("sbm_blocks must be an integer >= 2")
        for value in (self.ws_rewire_probability, self.sbm_within_share):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Topology probabilities must lie in [0, 1]")

    def parameters(self, n, m):
        if type(n) is not int or type(m) is not int or not 1 <= m < n:
            raise ValueError("Require integer 1 <= network_m < agent_count")
        target = m * (n - m)
        result = {**asdict(self), "agent_count": n, "network_m": m,
                  "ba_reference_edges": target, "expected_edges": target}
        if self.family == "er":
            result["er_probability"] = target / (n * (n - 1) / 2)
        elif self.family == "ws":
            if self.ws_k >= n:
                raise ValueError("ws_k must be below agent_count; no automatic clipping")
            result["expected_edges"] = n * self.ws_k // 2
        elif self.family == "sbm":
            if self.sbm_blocks > n:
                raise ValueError("SBM requires nonempty blocks")
            sizes = [n // self.sbm_blocks + (i < n % self.sbm_blocks)
                     for i in range(self.sbm_blocks)]
            within = sum(s * (s - 1) // 2 for s in sizes)
            between = n * (n - 1) // 2 - within
            if not within or not between:
                raise ValueError("SBM requires within- and between-block pairs")
            p_in = target * self.sbm_within_share / within
            p_out = target * (1 - self.sbm_within_share) / between
            if not 0 <= p_in <= 1 or not 0 <= p_out <= 1:
                raise ValueError("SBM edge target is infeasible; no probability clipping")
            result.update(block_sizes=sizes, sbm_p_in=p_in, sbm_p_out=p_out)
        return result


def undirected_graph(n, m, seed, topology):
    params = topology.parameters(n, m)
    if topology.family == "ba":
        return nx.barabasi_albert_graph(n, m, seed=seed)
    if topology.family == "er":
        return nx.erdos_renyi_graph(n, params["er_probability"], seed=seed)
    if topology.family == "ws":
        return nx.watts_strogatz_graph(n, topology.ws_k, topology.ws_rewire_probability, seed=seed)
    probabilities = [[params["sbm_p_in"] if i == j else params["sbm_p_out"]
                      for j in range(topology.sbm_blocks)] for i in range(topology.sbm_blocks)]
    return nx.stochastic_block_model(params["block_sizes"], probabilities, seed=seed,
                                    directed=False, selfloops=False)
