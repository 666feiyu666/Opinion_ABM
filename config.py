from __future__ import annotations

from copy import deepcopy

DEFAULT_SEED = 42

DEFAULT_PARAMS = {
    # Population / network
    "N": 1000,
    "m_BA": 3,
    "leader_in_degree_threshold": 20,
    "T_rounds": 80,
    # Initial opinion distribution
    "opinion_mean": 0.0,
    "opinion_std": 0.35,
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
    "alpha0": -2.0,
    "alpha1": 1.8,
    "alpha2": 1.2,
    "alpha3": 0.25,
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
    "gamma_R": 0.010,       
    "gamma_A": 0.008,       
    "gamma_B": 0.012,       
    "gamma_R_L": 0.006,     
    "gamma_A_L": 0.004,
    "gamma_B_L": 0.006,
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

    # ==========================================
    # Attention / Interest decay (新增参数)
    # ==========================================
    "base_create_prob": 1.0,   # 基准发帖概率（如果想降低全周期的发帖量，可以调低，如 0.8）
    "attention_decay": 0.05,   # 注意力衰减指数（值越大衰减越快，可根据需要调整）
}


def make_params(overrides: dict | None = None) -> dict:
    params = deepcopy(DEFAULT_PARAMS)
    if overrides:
        params.update(overrides)
    return params
