from __future__ import annotations

from utils import clip_opinion, sign_opinion
import numpy as np


def update_opinions(agents, params: dict):
    agents = agents.copy()
    agents["o_t1"] = agents["o_t"].copy()

    for i, row in agents.iterrows():
        direction = row["s_t"]
        if direction == 1:
            reinforce = np.log1p(row["M_pC_t"] + row["M_pT_t"])
            attenuate = np.log1p(row["M_nC_t"])
            backfire  = np.log1p(row["M_nT_t"])
        else:
            reinforce = np.log1p(row["M_nC_t"] + row["M_nT_t"])
            attenuate = np.log1p(row["M_pC_t"])
            backfire  = np.log1p(row["M_pT_t"])

        damping_factor = 1.0 - abs(row["o_t"])
        
        if row["L"] == 0:
            delta = (
                params["gamma_R"] * reinforce * direction
                - params["gamma_A"] * attenuate * direction
                + params["gamma_B"] * backfire * direction
            )
        else:
            delta = (
                params["gamma_R_L"] * reinforce * direction
                - params["gamma_A_L"] * attenuate * direction
                + params["gamma_B_L"] * backfire * direction
            )

        opinion_new = row["o_t"] + damping_factor * delta
        
        agents.at[i, "o_t1"] = clip_opinion(opinion_new)

    agents["s_t1"] = agents["o_t1"].apply(sign_opinion)
    return agents
