from __future__ import annotations

import numpy as np

from utils import clip_opinion, sample_stance_from_opinion


def _zone_for_opinion(opinion: float, tolerance_threshold: float) -> str:
    return "in" if abs(opinion) < tolerance_threshold else "out"


def _coefficient_set(params: dict, is_leader: bool, zone: str) -> tuple[float, float, float, float]:
    suffix = "_L" if is_leader else ""
    return (
        params[f"omega_pC_{zone}{suffix}"],
        params[f"omega_pT_{zone}{suffix}"],
        params[f"omega_nC_{zone}{suffix}"],
        params[f"omega_nT_{zone}{suffix}"],
    )


def update_opinions(agents, params: dict, rng: np.random.Generator):
    agents = agents.copy()
    agents["o_t1"] = agents["o_t"].copy()
    agents["s_t1"] = agents["s_t"].copy()

    tolerance_threshold = params.get("tolerance_threshold", 0.00)
    base_tau_env = params.get("tau_env", 0.5)
    max_confidence = params.get("max_confidence", 50.0)
    eta_expression = params.get("eta_expression", 1.0)

    for i, row in agents.iterrows():
        posts_read = row["M_pC_t"] + row["M_pT_t"] + row["M_nC_t"] + row["M_nT_t"]

        # Aggregate exposures remain count-based, but are log-saturated before use.
        exposure_pC = np.log1p(row["M_pC_t"])
        exposure_pT = np.log1p(row["M_pT_t"])
        exposure_nC = np.log1p(row["M_nC_t"])
        exposure_nT = np.log1p(row["M_nT_t"])

        zone = _zone_for_opinion(row["o_t"], tolerance_threshold)
        omega_pC, omega_pT, omega_nC, omega_nT = _coefficient_set(
            params,
            is_leader=bool(row["L"]),
            zone=zone,
        )

        # The current expressed stance anchors the signed push from all exposure types.
        delta = row["s_t"] * (
            omega_pC * exposure_pC
            + omega_pT * exposure_pT
            + omega_nC * exposure_nC
            + omega_nT * exposure_nT
        )

        damping_factor = 1.0 - 0.5 * abs(row["o_t"])
        tau_env = base_tau_env * np.log1p(posts_read)
        tau_t = float(row["confidence"])
        bayesian_weight = tau_env / (tau_t + tau_env) if (tau_t + tau_env) > 0 else 0.0

        opinion_new = row["o_t"] + damping_factor * bayesian_weight * delta
        opinion_new = clip_opinion(opinion_new)
        confidence_new = min(tau_t + tau_env, max_confidence)

        agents.at[i, "o_t1"] = opinion_new
        agents.at[i, "confidence"] = confidence_new
        agents.at[i, "s_t1"] = sample_stance_from_opinion(
            opinion=opinion_new,
            confidence=confidence_new,
            eta_expression=eta_expression,
            rng=rng,
        )

    return agents
