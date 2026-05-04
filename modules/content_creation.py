from __future__ import annotations

import math

import numpy as np


def _warmup_scale(current_round: int, warmup_rounds: float, floor: float) -> float:
    if warmup_rounds <= 0:
        return 1.0

    progress = min(max((current_round - 1) / warmup_rounds, 0.0), 1.0)
    return float(floor + (1.0 - floor) * progress)


def create_posts(agents, params: dict, rng: np.random.Generator, current_round: int = 1):
    agents = agents.copy()
    posts = {}

    beta_temp = params.get("temperature_C", 1.0)
    attention_decay = params.get("attention_decay", 0.05)
    base_create_prob = params.get("base_create_prob", 1.0)
    creation_warmup = _warmup_scale(
        current_round=current_round,
        warmup_rounds=params.get("creation_warmup_rounds", 0),
        floor=params.get("creation_warmup_floor", 1.0),
    )
    stance_feedback_scale = _warmup_scale(
        current_round=current_round,
        warmup_rounds=params.get("stance_feedback_warmup_rounds", 0),
        floor=params.get("stance_feedback_floor", 1.0),
    )
    style_feedback_scale = _warmup_scale(
        current_round=current_round,
        warmup_rounds=params.get("style_feedback_warmup_rounds", 0),
        floor=params.get("style_feedback_floor", 1.0),
    )

    current_prob = (
        base_create_prob
        * creation_warmup
        * math.exp(-attention_decay * (current_round - 1))
    )

    for i, row in agents.iterrows():
        if row["O_t"] != 1:
            continue

        if rng.random() > current_prob:
            agents.at[i, "C_t"] = 0
            continue

        eps = rng.normal(0, params["epsilon_std"])

        log_pC = np.log1p(row["M_pC_prev"])
        log_pT = np.log1p(row["M_pT_prev"])
        log_nC = np.log1p(row["M_nC_prev"])
        log_nT = np.log1p(row["M_nT_prev"])

        current_opinion = row["o_t"]
        tau_t = float(row["tau_t"])

        stance_social_plus = (
            params["beta1"] * log_pC
            + params["beta2"] * log_pT
            - params["beta3"] * log_nC
            - params["beta4"] * log_nT
        )
        stance_social_minus = (
            params["beta1"] * log_nC
            + params["beta2"] * log_nT
            - params["beta3"] * log_pC
            - params["beta4"] * log_pT
        )
        u_plus = (
            params["alpha_B"] * tau_t * max(current_opinion, 0.0)
            + stance_feedback_scale * stance_social_plus
            - params["c0"]
            + eps
        )
        u_minus = (
            params["alpha_B"] * tau_t * max(-current_opinion, 0.0)
            + stance_feedback_scale * stance_social_minus
            - params["c0"]
            + eps
        )

        if max(u_plus, u_minus) < 0:
            agents.at[i, "C_t"] = 0
            continue

        if u_plus >= 0 and u_minus >= 0 and abs(u_plus - u_minus) <= params["epsilon_ambiguity"]:
            agents.at[i, "C_t"] = 0
            continue

        agents.at[i, "C_t"] = 1

        beta_stance_temp = params.get("temperature_stance", 1.0)
        diff_u = np.clip((u_plus - u_minus) / beta_stance_temp, -500, 500)
        p_u = 1.0 / (1.0 + np.exp(-diff_u))

        kappa_t = params["kappa"] * tau_t
        alpha_t = 1.0 + kappa_t * p_u
        beta_t = 1.0 + kappa_t * (1.0 - p_u)

        pi_t = alpha_t / (alpha_t + beta_t)
        stance = 1 if rng.random() < pi_t else -1

        eta_c = rng.normal(0, params["eta_C_std"])
        eta_t = rng.normal(0, params["eta_T_std"])

        if stance == 1:
            style_social_c = (
                params["gamma1"] * log_pC
                - params["gamma2"] * log_pT
                - params["gamma3"] * log_nT
            )
            style_social_t = (
                params["delta1"] * log_pT
                + params["delta2"] * log_nT
                - params["delta4"] * log_pC
            )
            v_c = (
                params["gamma0"]
                + style_feedback_scale * style_social_c
                - params["gamma4"] * abs(row["o_t"])
                - params["c_C"]
                + eta_c
            )
            v_t = (
                params["delta0"]
                + style_feedback_scale * style_social_t
                + params["delta3"] * abs(row["o_t"])
                - params["c_T"]
                + eta_t
            )
        else:
            style_social_c = (
                params["gamma1"] * log_nC
                - params["gamma2"] * log_nT
                - params["gamma3"] * log_pT
            )
            style_social_t = (
                params["delta1"] * log_nT
                + params["delta2"] * log_pT
                - params["delta4"] * log_nC
            )
            v_c = (
                params["gamma0"]
                + style_feedback_scale * style_social_c
                - params["gamma4"] * abs(row["o_t"])
                - params["c_C"]
                + eta_c
            )
            v_t = (
                params["delta0"]
                + style_feedback_scale * style_social_t
                + params["delta3"] * abs(row["o_t"])
                - params["c_T"]
                + eta_t
            )

        diff_v = np.clip((v_c - v_t) / beta_temp, -500, 500)
        prob_C = 1.0 / (1.0 + np.exp(-diff_v))
        style = "C" if rng.random() < prob_C else "T"

        posts[i] = {
            "creator": i,
            "x": stance,
            "y": style,
            "q": int(row["L"]),
        }

    return agents, posts
