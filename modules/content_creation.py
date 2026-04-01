from __future__ import annotations

import numpy as np


def create_posts(agents, params: dict, rng: np.random.Generator):
    agents = agents.copy()
    posts = {}

    for i, row in agents.iterrows():
        if row["O_t"] != 1:
            continue

        eps = rng.normal(0, params["epsilon_std"])
        u_plus = (
            params["alpha_B"] * max(row["m_t"], 0.0)
            + params["beta1"] * row["M_pC_prev"]
            + params["beta2"] * row["M_pT_prev"]
            - params["beta3"] * row["M_nC_prev"]
            - params["beta4"] * row["M_nT_prev"]
            - params["c0"]
            + eps
        )
        u_minus = (
            params["alpha_B"] * max(-row["m_t"], 0.0)
            + params["beta1"] * row["M_nC_prev"]
            + params["beta2"] * row["M_nT_prev"]
            - params["beta3"] * row["M_pC_prev"]
            - params["beta4"] * row["M_pT_prev"]
            - params["c0"]
            + eps
        )

        if max(u_plus, u_minus) < 0:
            agents.at[i, "C_t"] = 0
            continue

        agents.at[i, "C_t"] = 1
        stance = 1 if u_plus >= u_minus else -1

        eta_c = rng.normal(0, params["eta_C_std"])
        eta_t = rng.normal(0, params["eta_T_std"])

        if stance == 1:
            v_c = (
                params["gamma0"]
                + params["gamma1"] * row["M_pC_prev"]
                - params["gamma2"] * row["M_pT_prev"]
                - params["gamma3"] * row["M_nT_prev"]
                - params["gamma4"] * abs(row["o_t"])
                - params["c_C"]
                + eta_c
            )
            v_t = (
                params["delta0"]
                + params["delta1"] * row["M_pT_prev"]
                + params["delta2"] * row["M_nT_prev"]
                + params["delta3"] * abs(row["o_t"])
                - params["delta4"] * row["M_pC_prev"]
                - params["c_T"]
                + eta_t
            )
        else:
            v_c = (
                params["gamma0"]
                + params["gamma1"] * row["M_nC_prev"]
                - params["gamma2"] * row["M_nT_prev"]
                - params["gamma3"] * row["M_pT_prev"]
                - params["gamma4"] * abs(row["o_t"])
                - params["c_C"]
                + eta_c
            )
            v_t = (
                params["delta0"]
                + params["delta1"] * row["M_nT_prev"]
                + params["delta2"] * row["M_pT_prev"]
                + params["delta3"] * abs(row["o_t"])
                - params["delta4"] * row["M_nC_prev"]
                - params["c_T"]
                + eta_t
            )

        style = "C" if v_c >= v_t else "T"
        posts[i] = {
            "creator": i,
            "x": stance,
            "y": style,
            "q": int(row["L"]),
        }

    return agents, posts
