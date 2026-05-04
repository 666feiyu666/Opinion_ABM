from __future__ import annotations

import numpy as np

from utils import clip_opinion, sample_stance_from_opinion


def _response_coefficients(params: dict, is_leader: bool) -> tuple[float, float, float, float]:
    suffix = "_L" if is_leader else ""
    return (
        params[f"omega_pC_out{suffix}"],
        params[f"omega_pT_out{suffix}"],
        params[f"omega_nC_out{suffix}"],
        params[f"omega_nT_out{suffix}"],
    )


def _relative_exposures(row) -> tuple[float, float, float, float]:
    if row["s_t"] >= 0:
        aligned_constructive = np.log1p(row["M_pC_t"])
        aligned_toxic = np.log1p(row["M_pT_t"])
        opposite_constructive = np.log1p(row["M_nC_t"])
        opposite_toxic = np.log1p(row["M_nT_t"])
    else:
        aligned_constructive = np.log1p(row["M_nC_t"])
        aligned_toxic = np.log1p(row["M_nT_t"])
        opposite_constructive = np.log1p(row["M_pC_t"])
        opposite_toxic = np.log1p(row["M_pT_t"])

    return (
        aligned_constructive,
        aligned_toxic,
        opposite_constructive,
        opposite_toxic,
    )


def update_opinions(agents, params: dict, rng: np.random.Generator):
    agents = agents.copy()
    agents["o_t1"] = agents["o_t"].copy()
    agents["s_t1"] = agents["s_t"].copy()

    tau_max = params["tau_max"]
    tau_env_0 = params["tau_env_0"]
    theta_conf = params.get("theta_conf", 0.5)
    eta_expression = params.get("eta_expression", 1.0)

    agents["tau_t"] = np.clip(agents["tau_t"], 0.1, tau_max)
    agents["tau_t1"] = agents["tau_t"].copy()

    for i, row in agents.iterrows():
        exposure_pC, exposure_pT, exposure_nC, exposure_nT = _relative_exposures(row)
        omega_pC, omega_pT, omega_nC, omega_nT = _response_coefficients(
            params,
            is_leader=bool(row["L"]),
        )

        reinforce = row["s_t"] * (omega_pC * exposure_pC + omega_pT * exposure_pT)
        attenuate = row["s_t"] * (omega_nC * exposure_nC)
        backfire = row["s_t"] * (omega_nT * exposure_nT)
        delta = reinforce + attenuate + backfire

        damping_factor = 1.0 - 0.5 * abs(row["o_t"])

        N_pos = row["M_pC_t"] + row["M_pT_t"]
        N_neg = row["M_nC_t"] + row["M_nT_t"]
        N_total = N_pos + N_neg
        R_clarity = abs(N_pos - N_neg) / N_total if N_total > 0 else 0.0

        tau_env = tau_env_0 * np.log1p(N_total)
        tau_i = float(row["tau_t"])
        tau_new = tau_i + tau_env * (R_clarity - theta_conf)
        tau_new = float(np.clip(tau_new, 0.1, tau_max))
        agents.at[i, "tau_t1"] = tau_new

        W_i = tau_env / (tau_i + tau_env) if tau_env > 0 else 0.0
        opinion_new = clip_opinion(row["o_t"] + damping_factor * W_i * delta)

        agents.at[i, "o_t1"] = opinion_new
        agents.at[i, "s_t1"] = sample_stance_from_opinion(
            opinion=opinion_new,
            confidence=tau_new,
            eta_expression=eta_expression,
            rng=rng,
        )

    agents["tau_t"] = agents["tau_t1"]
    return agents
