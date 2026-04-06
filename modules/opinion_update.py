from __future__ import annotations

import numpy as np
from utils import clip_opinion, sign_opinion


def update_opinions(agents, params: dict):
    agents = agents.copy()
    agents["o_t1"] = agents["o_t"].copy()

    # 获取卷入度极点阈值（如果在 config.py 中没有定义，则默认为 0.3）
    inv_threshold = params.get("involvement_threshold", 0.5)

    # 设定环境信号的基础精度（可以作为一个超参，比如 0.5）
    # tau_env 越大，代表每次看到的信息说服力越强
    base_tau_env = params.get("tau_env", 0.5)

    for i, row in agents.iterrows():
        direction = row["s_t"]

        # 统计本轮阅读的帖子总数
        posts_read = row["M_pC_t"] + row["M_pT_t"] + row["M_nC_t"] + row["M_nT_t"]
        
        # 1. 认知饱和机制（对数衰减）
        if direction == 1:
            reinforce = np.log1p(row["M_pC_t"] + row["M_pT_t"])
            attenuate = np.log1p(row["M_nC_t"])
            backfire  = np.log1p(row["M_nT_t"])
        else:
            reinforce = np.log1p(row["M_nC_t"] + row["M_nT_t"])
            attenuate = np.log1p(row["M_pC_t"])
            backfire  = np.log1p(row["M_pT_t"])

        damping_factor = 1.0 - abs(row["o_t"])
        
        # 2. 获取基础的影响力参数
        if row["L"] == 0:
            base_gamma_R = params["gamma_R"]
            base_gamma_A = params["gamma_A"]
            base_gamma_B = params["gamma_B"]
        else:
            base_gamma_R = params["gamma_R_L"]
            base_gamma_A = params["gamma_A_L"]
            base_gamma_B = params["gamma_B_L"]

        # ==================== 核心新增：跨越极点的 SJT 机制 ====================
        involvement = abs(row["o_t"])
        
        if involvement < inv_threshold:
            # 尚未跨过极点（温和派）：
            # - 关闭逆火效应（拥有宽广的非承诺域，不轻易被激怒）
            # - 放大回归中立的拉力（更容易被建设性意见拉回）
            actual_gamma_B = 0.0
            actual_gamma_A = base_gamma_A * 1.5 
            actual_gamma_R = base_gamma_R
        else:
            # 已经跨过极点（极端派）：
            # - 逆火效应全面开启，加速极化
            actual_gamma_B = base_gamma_B
            actual_gamma_A = base_gamma_A
            actual_gamma_R = base_gamma_R
        # =========================================================================

        # 计算增量（使用动态评估后的 Gamma）
        delta = (
            actual_gamma_R * reinforce * direction
            - actual_gamma_A * attenuate * direction
            + actual_gamma_B * backfire * direction
        )

        # ==================== 核心新增：贝叶斯精度加权 ====================
        # 本轮环境信号的精度与阅读量对数正相关（读得越多，信号越强）
        tau_env = base_tau_env * np.log1p(posts_read)
        
        # 提取当前确信度
        tau_t = row["confidence"]
        
        # 计算贝叶斯更新权重 W
        # 如果这是第一轮(tau_t=1)，W 较大；如果是第100轮(tau_t很大)，W 极小
        bayesian_weight = tau_env / (tau_t + tau_env)
        
        # 结合原有的极端值阻尼，计算最终观点
        opinion_new = row["o_t"] + damping_factor * bayesian_weight * delta
        agents.at[i, "o_t1"] = clip_opinion(opinion_new)
        
        # 贝叶斯后验更新：确信度随阅读量增加
        # （可选）添加一个信心上限或衰减，防止完全绝对固化
        max_confidence = 50.0 
        new_confidence = min(tau_t + tau_env, max_confidence)
        agents.at[i, "confidence"] = new_confidence
        # =================================================================

    agents["s_t1"] = agents["o_t1"].apply(sign_opinion)
    return agents