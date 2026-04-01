from __future__ import annotations

from utils import sigmoid


def creator_evaluation_score(viewer: int, creator: int, post: dict, agents, params: dict):
    stance = post["x"]
    style = post["y"]
    viewer_direction = agents.at[viewer, "s_t1"]
    creator_is_leader = agents.at[creator, "L"]

    aligned = int(stance == viewer_direction)
    opposite = int(stance != viewer_direction)
    toxic = int(style == "T")
    constructive = int(style == "C")

    v_ij = (
        params["a1"] * int(aligned and constructive)
        + params["a2"] * int(aligned and toxic)
        - params["a3"] * int(opposite and constructive)
        - params["a4"] * int(opposite and toxic)
    )
    k_ij = (
        params["b_T"] * toxic
        + params["b_O"] * opposite
        + params["b_TO"] * int(toxic and opposite)
    )
    s_ij = (
        params["lambda_V"] * v_ij
        - params["lambda_K"] * k_ij
        + params["lambda_L"] * creator_is_leader
    )
    return v_ij, k_ij, s_ij


def adapt_network(g_current, agents, exposure_sets: dict, posts: dict, params: dict, rng):
    g_next = g_current.copy()
    n_agents = params["N"]

    for viewer in range(n_agents):
        if agents.at[viewer, "L"] == 1:
            continue
        if agents.at[viewer, "C_t"] == 1:
            continue

        encountered = {post["creator"] for post in exposure_sets[viewer]}
        for creator in encountered:
            post = posts[creator]
            _, _, score = creator_evaluation_score(viewer, creator, post, agents, params)

            if g_current.has_edge(viewer, creator):
                p_maintain = sigmoid(score)
                keep = int(rng.binomial(1, p_maintain))
                if keep == 0 and g_next.has_edge(viewer, creator):
                    g_next.remove_edge(viewer, creator)
            else:
                p_follow = sigmoid(score - params["theta_F"])
                add_follow = int(rng.binomial(1, p_follow))
                if add_follow == 1 and not g_next.has_edge(viewer, creator):
                    g_next.add_edge(viewer, creator)

    return g_next
