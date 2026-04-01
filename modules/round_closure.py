from __future__ import annotations

import numpy as np


def finalize_round(g_next, agents, exposure_sets: dict, posts: dict, params: dict):
    agents = agents.copy()
    n_agents = params["N"]

    agents["F_t1"] = [g_next.in_degree(i) for i in range(n_agents)]

    next_activity = []
    for viewer, row in agents.iterrows():
        exposure_size = len(exposure_sets[viewer])
        empty_indicator = 1 if exposure_size == 0 else 0
        activity_next = (
            (1 - params["rho_A"]) * row["A_t"]
            + params["rho_A"] * row["Abar"]
            + params["omega1"] * row["C_t"]
            + params["omega2"] * (exposure_size / params["N_E"])
            - params["omega3"] * empty_indicator
        )
        next_activity.append(float(np.clip(activity_next, 0.0, 1.0)))

    agents["A_t1"] = next_activity

    agents["o_t"] = agents["o_t1"]
    agents["s_t"] = agents["s_t1"]
    agents["A_t"] = agents["A_t1"]
    agents["F_t"] = agents["F_t1"]

    agents["M_pC_prev"] = agents["M_pC_t"]
    agents["M_pT_prev"] = agents["M_pT_t"]
    agents["M_nC_prev"] = agents["M_nC_t"]
    agents["M_nT_prev"] = agents["M_nT_t"]

    for i in range(n_agents):
        g_next.nodes[i]["opinion"] = agents.at[i, "o_t"]
        g_next.nodes[i]["leader"] = agents.at[i, "L"]

    summary = {
        "potential_originators": int(agents["O_t"].sum()),
        "actual_creators": int(agents["C_t"].sum()),
        "constructive_posts": int(sum(1 for post in posts.values() if post["y"] == "C")),
        "toxic_posts": int(sum(1 for post in posts.values() if post["y"] == "T")),
        "support_posts": int(sum(1 for post in posts.values() if post["x"] == 1)),
        "oppose_posts": int(sum(1 for post in posts.values() if post["x"] == -1)),
        "avg_exposure_size": float(np.mean([len(exposure_sets[i]) for i in range(n_agents)])),
        "mean_opinion": float(agents["o_t"].mean()),
        "std_opinion": float(agents["o_t"].std()),
        "edge_count": int(g_next.number_of_edges()),
    }
    return g_next, agents, summary
