from __future__ import annotations

from copy import deepcopy

DEFAULT_SEED = 42

NOTEBOOK_BASELINE_OVERRIDES = {
    "tolerance_threshold": 0.00,
    "involvement_threshold": 0.60,
    # Wider initial opinion spread.
    "opinion_std": 0.5,
    # Lower confidence freezing so polarization can still move.
    "tau_env": 0.90,
    "max_confidence": 15.0,
    # Stronger reinforcement / backfire outside the tolerance zone.
    "omega_pC_out": 0.12,
    "omega_pT_out": 0.04,
    "omega_nT_out": 0.10,
    "omega_nC_out": 0.00,
    # Higher exposure volume.
    "p_O": 0.04,
    "beta2_diff": 1.00,
    "max_read_capacity": 12,
    # Looser warmup, matching the notebook baseline.
    "creation_warmup_rounds": 2,
    "stance_feedback_warmup_rounds": 4,
    "style_feedback_warmup_rounds": 4,
    "creation_warmup_floor": 0.80,
    "stance_feedback_floor": 0.70,
    "style_feedback_floor": 0.70,
    # Easier echo-chamber rewiring.
    "theta_F": 2.8,
    "a4": 1.6,
    "b_O": 0.6,
    "b_TO": 1.0,
    # Toxicity-first involvement dynamics for the notebook baseline.
    "involvement_toxic_gain": 0.16,
    "involvement_exposure_gain": 0.02,
    "involvement_decay": 0.10,
}

SIMPLE_LEADER_INFLUENCE_OVERRIDES = {
    "tolerance_threshold": 0.00,
    "involvement_threshold": 0.00,
    # Calibrate origination so ordinary users start near 0.1 while leaders
    # remain a clearly stronger content source around 0.3-0.4.
    "alpha0": -3.35,
    "alpha2": 1.10,
    "originator_prob_cap": 0.40,
    # Leave room for persuasion in the baseline: weaker backfire, stronger
    # constructive pullback, and a materially higher cost for toxic posting.
    "omega_nT_out": 0.02,
    "omega_nT_out_L": 0.015,
    "omega_nC_out": -0.04,
    "omega_nC_out_L": -0.025,
    "delta1": 0.25,
    "delta2": 0.15,
    "delta3": 0.15,
    "c_T": 0.34,
    # Freeze involvement so the baseline isolates leader-direction effects.
    "e_init_mean": 1.00,
    "e_init_std": 0.00,
    "e_L_init_mean": 1.00,
    "e_L_init_std": 0.00,
    "involvement_toxic_gain": 0.00,
    "involvement_exposure_gain": 0.00,
    "involvement_decay": 0.00,
}

DEFAULT_PARAMS = {
    # Population / network
    "N": 1000,
    "m_BA": 3,
    "network_topology": "BA",
    "leader_share": None,
    "leader_selection_method": "threshold",
    "leader_in_degree_threshold": 15,
    "ws_k": 6,
    "ws_rewire_prob": 0.10,
    "sbm_num_blocks": 4,
    "sbm_within_share": 0.70,
    "T_rounds": 50,
    # Initial opinion distribution
    "opinion_mean": 0.0,
    "opinion_std": 0.5,
    # Activity
    "Abar_low": 0.2,
    "Abar_high": 0.8,
    "rho_A": 0.6,
    "omega1": 0.10,
    "omega2": 0.20,
    "omega3": 0.05,
    "N_E": 10.0,
    # Opinion-expression mapping
    "kappa": 2.0,
    # Content-chain origination
    "alpha0": -2.25,
    "alpha1": 1.8,
    "alpha2": 1.2,
    "alpha3": 0.25,
    "originator_prob_cap": 0.30,
    # Stance-choice utility
    "alpha_B": 1.5,
    "beta1": 0.60,
    "beta2": 0.40,
    "beta3": 0.50,
    "beta4": 0.70,
    "c0": 0.55,
    "epsilon_std": 0.15,
    "epsilon_ambiguity": 0.5, 
    # Style-choice utility: constructive
    "gamma0": 0.15,
    "gamma1": 0.55,
    "gamma2": 0.35,
    "gamma3": 0.45,
    "gamma4": 0.25,
    "c_C": 0.10,
    "eta_C_std": 0.12,
    # Style-choice utility: toxic
    "delta0": 0.05,
    "delta1": 0.55,
    "delta2": 0.45,
    "delta3": 0.40,
    "delta4": 0.30,
    "c_T": 0.18,
    "eta_T_std": 0.12,
    # Diffusion
    "p_O": 0.02,
    "beta0_diff": -3.2,
    "beta1_diff": 0.80,
    "beta2_diff": 0.70,
    "max_read_capacity":8,
    # Exposure weights
    "w_o": 1.0,
    "w_l": 1.5,
    # Opinion updating
    "tolerance_threshold": 0.00,
    "tau_env": 0.50,
    "eta_expression": 1.00,
    "max_confidence": 50.0,
    # Inside tolerance range: weaker directional reinforcement, stronger constructive pullback.
    "omega_pC_in": 0.050,
    "omega_pT_in": 0.010,
    "omega_nC_in": -0.040,
    "omega_nT_in": 0.010,
    # Outside tolerance range: stronger reinforcement and stronger backfire.
    "omega_pC_out": 0.080,
    "omega_pT_out": 0.020,
    "omega_nC_out": -0.005,
    "omega_nT_out": 0.050,
    # Leaders follow the same structure with a lower response magnitude.
    "omega_pC_in_L": 0.035,
    "omega_pT_in_L": 0.008,
    "omega_nC_in_L": -0.028,
    "omega_nT_in_L": 0.008,
    "omega_pC_out_L": 0.055,
    "omega_pT_out_L": 0.015,
    "omega_nC_out_L": -0.003,
    "omega_nT_out_L": 0.035,
    # Creator evaluation
    "a1": 1.0,
    "a2": 0.6,
    "a3": 0.7,
    "a4": 1.2,
    "b_T": 0.5,
    "b_O": 0.4,
    "b_TO": 0.8,
    "lambda_V": 0.8,   
    "lambda_K": 1.0,
    "lambda_L": 0.5,
    "theta_F": 3.5,
    "network_in_zone_contact_bonus": 0.30,
    "network_in_zone_opposite_constructive_bonus": 0.20,
    "network_in_zone_opposite_toxic_bonus": 0.15,
    "network_out_zone_opposite_penalty": 1.00,
    "network_out_zone_toxic_penalty": 0.80,
    "network_out_zone_disconnect_bias": 0.80,

    # ==========================================
    # Attention / Interest decay (新增参数)
    # ==========================================
    "base_create_prob": 1.0,   # 基准发帖概率（长期上限）
    "attention_decay": 0.005,   # 注意力衰减指数（值越大衰减越快，可根据需要调整）
    "creation_warmup_rounds": 8,
    "creation_warmup_floor": 0.40,
    "stance_feedback_warmup_rounds": 12,
    "stance_feedback_floor": 0.15,
    "style_feedback_warmup_rounds": 10,
    "style_feedback_floor": 0.30,
    "non_involved_creation_floor": 0.25,
    "non_involved_creation_shape": 1.5,
    "e_init_mean": 0.12,
    "e_init_std": 0.05,
    "e_L_init_mean": 0.90,
    "e_L_init_std": 0.05,
    "involvement_max": 1.0,
    "involvement_decay": 0.10,
    "involvement_toxic_gain": 0.22,
    "involvement_exposure_gain": 0.04,
    # Bayesian confidence / clarity dynamics
    "tau_max": 10.0,          # 置信度的绝对上限
    "tau_env_0": 0.5,         # 环境基础精度系数
    "theta_conf": 0.5,        # 清晰度指数阈值，决定环境是一边倒还是撕裂
    "tau_init_mean": 2.0,     # 大众用户的初始置信度均值
    "tau_init_std": 0.5,      # 大众用户的初始置信度标准差
    "tau_L_init_mean": 8.0,   # 意见领袖的极高初始置信度均值
    "tau_L_init_std": 1.0,    # 意见领袖的极高初始置信度标准差
    "leader_opinion_mode": "balanced",
}


def make_params(overrides: dict | None = None) -> dict:
    params = deepcopy(DEFAULT_PARAMS)
    if overrides:
        params.update(overrides)
    return params


def make_notebook_baseline_params(overrides: dict | None = None) -> dict:
    params = make_params(NOTEBOOK_BASELINE_OVERRIDES)
    if overrides:
        params.update(overrides)
    return params


def make_simple_leader_influence_params(
    leader_opinion_mode: str = "positive",
    overrides: dict | None = None,
) -> dict:
    params = make_notebook_baseline_params(SIMPLE_LEADER_INFLUENCE_OVERRIDES)
    params["leader_opinion_mode"] = leader_opinion_mode
    if overrides:
        params.update(overrides)
    return params
