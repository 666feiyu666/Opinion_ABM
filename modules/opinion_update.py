from __future__ import annotations

import numpy as np
from utils import clip_opinion, sign_opinion


def update_opinions(agents, params: dict):
    agents = agents.copy()
    agents["o_t1"] = agents["o_t"].copy()

    # 获取卷入度极点阈值（如果在 config.py 中没有定义，则默认为 0.3）
    inv_threshold = params.get("involvement_threshold", 0.3)

    for i, row in agents.iterrows():
        direction = row["s_t"]
        
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

        opinion_new = row["o_t"] + damping_factor * delta
        
        agents.at[i, "o_t1"] = clip_opinion(opinion_new)

    agents["s_t1"] = agents["o_t1"].apply(sign_opinion)
    return agents