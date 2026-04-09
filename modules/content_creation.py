from __future__ import annotations

import math
import numpy as np


def _warmup_scale(current_round: int, warmup_rounds: float, floor: float) -> float:
    if warmup_rounds <= 0:
        return 1.0

    progress = min(max((current_round - 1) / warmup_rounds, 0.0), 1.0)
    return float(floor + (1.0 - floor) * progress)


def _non_involved_creation_scale(
    opinion: float,
    tolerance_threshold: float,
    floor: float,
    shape: float,
) -> float:
    if tolerance_threshold <= 0:
        return 1.0

    distance = abs(float(opinion))
    if distance >= tolerance_threshold:
        return 1.0

    normalized_distance = distance / tolerance_threshold
    curved_progress = normalized_distance ** max(shape, 1e-6)
    return float(floor + (1.0 - floor) * curved_progress)


def create_posts(agents, params: dict, rng: np.random.Generator, current_round: int = 1):
    agents = agents.copy()
    posts = {}
    
    # ==================== 1. 读取参数 ====================
    # 引入温度参数
    beta_temp = params.get("temperature_C", 1.0) # 控制 C/T 转变平滑度
    beta_temp_stance = params.get("temperature_S", 1.0) # 控制 Support/Oppose 转变平滑度
    
    # 引入注意力/兴趣衰减参数 (新增)
    attention_decay = params.get("attention_decay", 0.05)
    base_create_prob = params.get("base_create_prob", 1.0)
    tolerance_threshold = params.get("tolerance_threshold", 0.0)
    non_involved_creation_floor = params.get("non_involved_creation_floor", 1.0)
    non_involved_creation_shape = params.get("non_involved_creation_shape", 1.0)
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
    
    # ==================== 2. 计算宏观衰减 ====================
    # 前几轮只释放一部分创作活性，避免创作者数量过快冲高；
    # 在此基础上再叠加全局注意力衰减。
    current_prob = (
        base_create_prob
        * creation_warmup
        * math.exp(-attention_decay * (current_round - 1))
    )

    for i, row in agents.iterrows():
        if row["O_t"] != 1:
            continue

        # ==================== 3. 疲劳/兴趣衰减判定 ====================
        # 非卷入区间内，越接近中立点，实际发帖意愿越弱。
        involvement_scale = _non_involved_creation_scale(
            opinion=row["o_t"],
            tolerance_threshold=tolerance_threshold,
            floor=non_involved_creation_floor,
            shape=non_involved_creation_shape,
        )
        effective_create_prob = current_prob * involvement_scale

        # 如果随机数大于当前衰减后的概率，说明该 Agent 本轮不活跃，跳过计算
        if rng.random() > effective_create_prob:
            agents.at[i, "C_t"] = 0
            continue

        eps = rng.normal(0, params["epsilon_std"])
        
        # 使用 np.log1p 将绝对阅读量转化为对数级刺激
        log_pC = np.log1p(row["M_pC_prev"])
        log_pT = np.log1p(row["M_pT_prev"])
        log_nC = np.log1p(row["M_nC_prev"])
        log_nT = np.log1p(row["M_nT_prev"])

        # 使用动态更新的观点 o_t 作为内部信仰锚点
        current_opinion = row["o_t"]

        # 发帖意愿（u）同样使用平滑后的 log 值
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
            params["alpha_B"] * max(current_opinion, 0.0)
            + stance_feedback_scale * stance_social_plus
            - params["c0"]
            + eps
        )
        u_minus = (
            params["alpha_B"] * max(-current_opinion, 0.0)
            + stance_feedback_scale * stance_social_minus
            - params["c0"]
            + eps
        )

        if max(u_plus, u_minus) < 0:
            agents.at[i, "C_t"] = 0
            continue

        if u_plus >= 0 and u_minus >= 0:
            if abs(u_plus - u_minus) <= params["epsilon_ambiguity"]:
                agents.at[i, "C_t"] = 0
                continue

        agents.at[i, "C_t"] = 1
        
        # ==================== 核心修复：立场(Stance)的概率化映射 ====================
        # 计算 Agent 产出 Support (+1) 帖子的概率
        diff_u = np.clip((u_plus - u_minus) / beta_temp_stance, -500, 500)
        prob_support = 1.0 / (1.0 + np.exp(-diff_u))
        
        # 抛硬币决定最终立场，允许“逆风输出”的少数派存在
        stance = 1 if rng.random() < prob_support else -1
        # =========================================================================

        eta_c = rng.normal(0, params["eta_C_std"])
        eta_t = rng.normal(0, params["eta_T_std"])

        # 计算内容风格的效用
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

        # 风格(Style)的概率化映射
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
