from __future__ import annotations

import numpy as np

from utils import sigmoid


def apply_origination(agents, params: dict, rng: np.random.Generator):
    agents = agents.copy()

    def compute_originator_probability(row):
        z_value = (
            params["alpha0"]
            + params["alpha1"] * row["A_t"]
            + params["alpha2"] * row["L"]
            + params["alpha3"] * np.log1p(row["F_t"])
        )
        return sigmoid(z_value)

    agents["pi_t"] = agents.apply(compute_originator_probability, axis=1)
    agents["O_t"] = rng.binomial(1, agents["pi_t"].to_numpy())
    return agents
