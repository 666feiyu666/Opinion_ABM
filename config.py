from __future__ import annotations

from copy import deepcopy

DEFAULT_SEED = 42

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
    "tau_init_mean": 2.0,
    "tau_init_std": 0.5,
    "tau_L_init_mean": 8.0,
    "tau_L_init_std": 1.0,
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
    "eta_expression": 1.00,
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
    "max_read_capacity": 8,
    # Exposure weights
    "w_o": 1.0,
    "w_l": 1.5,
    # Opinion updating
    "tau_max": 10.0,
    "tau_env_0": 0.5,
    "theta_conf": 0.5,
    "omega_pC_out": 0.080,
    "omega_pT_out": 0.020,
    "omega_nC_out": -0.005,
    "omega_nT_out": 0.050,
    "omega_pC_out_L": 0.055,
    "omega_pT_out_L": 0.015,
    "omega_nC_out_L": -0.003,
    "omega_nT_out_L": 0.035,
    # Creator evaluation / network adaptation
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
    "network_out_zone_opposite_penalty": 1.00,
    "network_out_zone_toxic_penalty": 0.80,
    "network_out_zone_disconnect_bias": 0.80,
    # Posting cadence
    "base_create_prob": 1.0,
    "attention_decay": 0.005,
    "creation_warmup_rounds": 8,
    "creation_warmup_floor": 0.40,
    "stance_feedback_warmup_rounds": 12,
    "stance_feedback_floor": 0.15,
    "style_feedback_warmup_rounds": 10,
    "style_feedback_floor": 0.30,
    "leader_opinion_mode": "balanced",
}

SIMPLE_LEADER_INFLUENCE_OVERRIDES = {
    # Calibrate origination so ordinary users start near 0.1 while leaders
    # remain a clearly stronger content source around 0.3-0.4.
    "alpha0": -3.35,
    "alpha2": 1.10,
    "originator_prob_cap": 0.40,
    # Leave room for persuasion in the thesis baseline.
    "omega_pC_out": 0.12,
    "omega_pT_out": 0.04,
    "omega_nT_out": 0.02,
    "omega_nT_out_L": 0.015,
    "omega_nC_out": -0.04,
    "omega_nC_out_L": -0.025,
    "delta1": 0.25,
    "delta2": 0.15,
    "delta3": 0.15,
    "c_T": 0.34,
    # Higher exposure volume and a shorter warmup match the thesis runs.
    "p_O": 0.04,
    "beta2_diff": 1.00,
    "max_read_capacity": 12,
    "creation_warmup_rounds": 2,
    "creation_warmup_floor": 0.80,
    "stance_feedback_warmup_rounds": 4,
    "stance_feedback_floor": 0.70,
    "style_feedback_warmup_rounds": 4,
    "style_feedback_floor": 0.70,
    # Easier echo-chamber rewiring.
    "theta_F": 2.8,
    "a4": 1.6,
    "b_O": 0.6,
    "b_TO": 1.0,
}


def make_params(overrides: dict | None = None) -> dict:
    params = deepcopy(DEFAULT_PARAMS)
    if overrides:
        params.update(overrides)
    return params


def make_simple_leader_influence_params(
    leader_opinion_mode: str = "positive",
    overrides: dict | None = None,
) -> dict:
    params = make_params(SIMPLE_LEADER_INFLUENCE_OVERRIDES)
    params["leader_opinion_mode"] = leader_opinion_mode
    if overrides:
        params.update(overrides)
    return params
