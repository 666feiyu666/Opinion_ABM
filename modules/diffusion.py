from __future__ import annotations

import numpy as np

from utils import sigmoid


def diffuse_posts(g_current, agents, blocks: dict, posts: dict, params: dict, rng):
    n_agents = params["N"]
    exposure_sets = {i: [] for i in range(n_agents)}
    exposure_matrix = {}

    for creator, post in posts.items():
        creator_is_leader = post["q"]
        creator_is_toxic = int(post["y"] == "T")
        followers = agents.at[creator, "F_t"]

        for viewer in range(n_agents):
            if viewer == creator:
                exposure_matrix[(viewer, creator)] = 0
                continue

            if creator in blocks.get(viewer, set()):
                exposure_matrix[(viewer, creator)] = 0
                continue

            if g_current.has_edge(viewer, creator):
                exposure_matrix[(viewer, creator)] = 1
                exposure_sets[viewer].append(post)
                continue

            if creator_is_leader == 0:
                p_exposure = params["p_O"]
            else:
                z_value = (
                    params["beta0_diff"]
                    + params["beta1_diff"] * np.log1p(followers)
                    + params["beta2_diff"] * creator_is_toxic
                )
                p_exposure = sigmoid(z_value)

            exposed = int(rng.binomial(1, p_exposure))
            exposure_matrix[(viewer, creator)] = exposed
            if exposed == 1:
                exposure_sets[viewer].append(post)

    agents = agents.copy()
    agents["M_pC_t"] = 0.0
    agents["M_pT_t"] = 0.0
    agents["M_nC_t"] = 0.0
    agents["M_nT_t"] = 0.0

    for viewer in range(n_agents):
        if len(exposure_sets[viewer]) > params.get("max_read_capacity", float('inf')):
            exposure_sets[viewer].sort(
                key=lambda p: agents.at[p["creator"], "F_t"], 
                reverse=True
            )
            
            dropped_posts = exposure_sets[viewer][params["max_read_capacity"]:]
            for post in dropped_posts:
                exposure_matrix[(viewer, post["creator"])] = 0
                
            exposure_sets[viewer] = exposure_sets[viewer][:params["max_read_capacity"]]
            
        m_pc = 0.0
        m_pt = 0.0
        m_nc = 0.0
        m_nt = 0.0

        for post in exposure_sets[viewer]:
            weight = params["w_l"] if post["q"] == 1 else params["w_o"]
            if post["x"] == 1 and post["y"] == "C":
                m_pc += weight
            elif post["x"] == 1 and post["y"] == "T":
                m_pt += weight
            elif post["x"] == -1 and post["y"] == "C":
                m_nc += weight
            elif post["x"] == -1 and post["y"] == "T":
                m_nt += weight

    return agents, exposure_sets, exposure_matrix
