from __future__ import annotations

import math
import numpy as np

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
    
    # ==================== 2. 计算宏观衰减 ====================
    # 计算当前轮次的基准发言概率 (指数衰减)
    # 第一轮 (current_round=1) 时指数为0，概率等于 base_create_prob
    current_prob = base_create_prob * math.exp(-attention_decay * (current_round - 1))

    for i, row in agents.iterrows():
        if row["O_t"] != 1:
            continue

        # ==================== 3. 疲劳/兴趣衰减判定 ====================
        # 如果随机数大于当前衰减后的概率，说明该 Agent 本轮不活跃，跳过计算
        if rng.random() > current_prob:
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
        u_plus = (
            params["alpha_B"] * max(current_opinion, 0.0)
            + params["beta1"] * log_pC
            + params["beta2"] * log_pT
            - params["beta3"] * log_nC
            - params["beta4"] * log_nT
            - params["c0"]
            + eps
        )
        u_minus = (
            params["alpha_B"] * max(-current_opinion, 0.0)
            + params["beta1"] * log_nC
            + params["beta2"] * log_nT
            - params["beta3"] * log_pC
            - params["beta4"] * log_pT
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
            v_c = (
                params["gamma0"]
                + params["gamma1"] * log_pC
                - params["gamma2"] * log_pT
                - params["gamma3"] * log_nT
                - params["gamma4"] * abs(row["o_t"])
                - params["c_C"]
                + eta_c
            )
            v_t = (
                params["delta0"]
                + params["delta1"] * log_pT
                + params["delta2"] * log_nT
                + params["delta3"] * abs(row["o_t"])
                - params["delta4"] * log_pC
                - params["c_T"]
                + eta_t
            )
        else:
            v_c = (
                params["gamma0"]
                + params["gamma1"] * log_nC
                - params["gamma2"] * log_nT
                - params["gamma3"] * log_pT
                - params["gamma4"] * abs(row["o_t"])
                - params["c_C"]
                + eta_c
            )
            v_t = (
                params["delta0"]
                + params["delta1"] * log_nT
                + params["delta2"] * log_pT
                + params["delta3"] * abs(row["o_t"])
                - params["delta4"] * log_nC
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