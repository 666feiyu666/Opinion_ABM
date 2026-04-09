from __future__ import annotations

from utils import sigmoid


def _viewer_zone(agents, viewer: int, params: dict) -> str:
    tolerance_threshold = params.get("tolerance_threshold", 0.0)
    if tolerance_threshold <= 0:
        return "out"
    return "in" if abs(float(agents.at[viewer, "o_t1"])) < tolerance_threshold else "out"


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


def _zone_adjusted_score(
    viewer: int,
    post: dict,
    base_score: float,
    has_edge: bool,
    agents,
    params: dict,
) -> float:
    zone = _viewer_zone(agents, viewer, params)
    viewer_direction = agents.at[viewer, "s_t1"]
    opposite = int(post["x"] != viewer_direction)
    constructive = int(post["y"] == "C")
    toxic = int(post["y"] == "T")

    score = float(base_score)

    if zone == "in":
        if has_edge:
            score += params.get("network_in_zone_contact_bonus", 0.0)
        if opposite and constructive:
            score += params.get("network_in_zone_opposite_constructive_bonus", 0.0)
        elif opposite and toxic:
            score += params.get("network_in_zone_opposite_toxic_bonus", 0.0)
        return score

    if opposite:
        score -= params.get("network_out_zone_opposite_penalty", 0.0)
        if toxic:
            score -= params.get("network_out_zone_toxic_penalty", 0.0)
        if has_edge:
            score -= params.get("network_out_zone_disconnect_bias", 0.0)

    return score


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
            _, _, base_score = creator_evaluation_score(viewer, creator, post, agents, params)

            if g_current.has_edge(viewer, creator):
                score = _zone_adjusted_score(
                    viewer=viewer,
                    post=post,
                    base_score=base_score,
                    has_edge=True,
                    agents=agents,
                    params=params,
                )
                p_maintain = sigmoid(score)
                keep = int(rng.binomial(1, p_maintain))
                if keep == 0 and g_next.has_edge(viewer, creator):
                    g_next.remove_edge(viewer, creator)
            else:
                score = _zone_adjusted_score(
                    viewer=viewer,
                    post=post,
                    base_score=base_score,
                    has_edge=False,
                    agents=agents,
                    params=params,
                )
                p_follow = sigmoid(score - params["theta_F"])
                add_follow = int(rng.binomial(1, p_follow))
                if add_follow == 1 and not g_next.has_edge(viewer, creator):
                    g_next.add_edge(viewer, creator)

    return g_next
