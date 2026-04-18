from __future__ import annotations

import numpy as np

from utils import clip_opinion, sample_stance_from_opinion


def _zone_for_involvement(involvement: float, involvement_threshold: float) -> str:
    if involvement_threshold <= 0:
        return "out"
    return "in" if involvement < involvement_threshold else "out"


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
    involvement_threshold = params.get(
        "involvement_threshold",
        params.get("tolerance_threshold", 0.00),
    )
    tau_max = params.get("tau_max", params.get("max_confidence", 10.0))
    tau_env_0 = params.get("tau_env_0", params.get("tau_env", 0.5))
    theta_conf = params.get("theta_conf", 0.5)
    eta_expression = params.get("eta_expression", 1.0)
    involvement_max = params.get("involvement_max", 1.0)
    involvement_decay = params.get("involvement_decay", 0.10)
    involvement_toxic_gain = params.get("involvement_toxic_gain", 0.22)
    involvement_exposure_gain = params.get("involvement_exposure_gain", 0.04)

    if "tau_t" not in agents.columns:
        if "confidence" in agents.columns:
            agents["tau_t"] = agents["confidence"].copy()
        else:
            agents["tau_t"] = 1.0
    if "e_t" not in agents.columns:
        agents["e_t"] = 1.0 if involvement_threshold <= 0 else 0.0

    agents["tau_t"] = np.clip(agents["tau_t"], 0.1, tau_max)
    agents["tau_t1"] = agents["tau_t"].copy()
    agents["e_t"] = np.clip(agents["e_t"], 0.0, involvement_max)
    agents["e_t1"] = agents["e_t"].copy()

    for i, row in agents.iterrows():
        # Aggregate exposures remain count-based, but are log-saturated before use.
        exposure_pC = np.log1p(row["M_pC_t"])
        exposure_pT = np.log1p(row["M_pT_t"])
        exposure_nC = np.log1p(row["M_nC_t"])
        exposure_nT = np.log1p(row["M_nT_t"])

        N_pos = row["M_pC_t"] + row["M_pT_t"]
        N_neg = row["M_nC_t"] + row["M_nT_t"]
        N_total = N_pos + N_neg
        N_toxic = row["M_pT_t"] + row["M_nT_t"]

        involvement_old = float(row["e_t"])
        toxic_share = (N_toxic / N_total) if N_total > 0 else 0.0
        exposure_density = min(1.0, N_total / max(params.get("max_read_capacity", 1), 1))
        activation = (
            involvement_toxic_gain * toxic_share
            + involvement_exposure_gain * exposure_density
        )
        involvement_new = (
            involvement_old
            + (involvement_max - involvement_old) * activation
            - involvement_decay * involvement_old
        )
        involvement_new = float(np.clip(involvement_new, 0.0, involvement_max))
        agents.at[i, "e_t1"] = involvement_new

        zone = _zone_for_involvement(involvement_new, involvement_threshold)
        omega_pC, omega_pT, omega_nC, omega_nT = _coefficient_set(
            params,
            is_leader=bool(row["L"]),
            zone=zone,
        )

        # 保留原有 SJT 对数饱和推力：同向曝光强化，异向建设性内容可能缓和，异向攻击性内容可能触发反弹。
        reinforce = row["s_t"] * (omega_pC * exposure_pC + omega_pT * exposure_pT)
        attenuate = row["s_t"] * (omega_nC * exposure_nC)
        backfire = row["s_t"] * (omega_nT * exposure_nT)
        delta = reinforce + attenuate + backfire

        damping_factor = 1.0 - 0.5 * abs(row["o_t"])
        if involvement_threshold > 0:
            involvement_gate = min(1.0, involvement_new / involvement_threshold)
        else:
            involvement_gate = 1.0

        # 清晰度指数衡量当前信息环境是否“一边倒”。
        # 越接近 1，说明接触到的信息越单边清晰；越接近 0，说明环境越撕裂混杂。
        if N_total > 0:
            R_clarity = abs(N_pos - N_neg) / N_total
        else:
            R_clarity = 0.0

        tau_env = tau_env_0 * np.log1p(N_total)
        multiplier = R_clarity - theta_conf

        tau_i = float(row["tau_t"])
        tau_new = tau_i + tau_env * multiplier
        tau_new = float(np.clip(tau_new, 0.1, tau_max))
        agents.at[i, "tau_t1"] = tau_new

        # 贝叶斯权重表示主体对外部信息的开放程度。
        # 当环境信号更“精确”时，外部证据在意见更新中占比更高；
        # 当主体自身置信度更高时，外部推动会被相对压低。
        if tau_env > 0:
            W_i = tau_env / (tau_i + tau_env)
        else:
            W_i = 0.0

        opinion_new = row["o_t"] + involvement_gate * damping_factor * W_i * delta
        opinion_new = clip_opinion(opinion_new)

        agents.at[i, "o_t1"] = opinion_new
        agents.at[i, "s_t1"] = sample_stance_from_opinion(
            opinion=opinion_new,
            confidence=tau_new,
            eta_expression=eta_expression,
            rng=rng,
        )

    # 兼容现有未改动模块：同时维护新旧置信度列。
    agents["tau_t"] = agents["tau_t1"]
    agents["confidence"] = agents["tau_t1"]
    agents["e_t"] = agents["e_t1"]

    return agents
