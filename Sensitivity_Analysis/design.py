from __future__ import annotations

import json

import pandas as pd

from config import DEFAULT_SEED

N_AGENTS = 1000
LEADER_MODES = ["balanced", "positive", "negative"]
TOPOLOGIES = ["BA", "SBM"]
BENCHMARK_LEADER_SHARE = 0.03
SHARE_LEVELS = [0.01, 0.03, 0.05]

PROFILE_SPECS = {
    "smoke": {
        "description": "Reduced integration check; not intended for thesis interpretation.",
        "seed_count": 1,
        "T_rounds": 2,
        "topologies": ["BA"],
        "leader_modes": LEADER_MODES,
        "core_perturbation_ids": ["nominal", "w_l_low", "w_l_high"],
        "share_perturbation_ids": ["nominal", "w_l_low", "w_l_high"],
    },
    "trial": {
        "description": "Complete experiment structure with fewer seeds and rounds.",
        "seed_count": 2,
        "T_rounds": 25,
    },
    "main": {
        "description": "Thesis robustness profile with ten random seeds and 50 rounds.",
        "seed_count": 10,
        "T_rounds": 50,
    },
}

PARAMETER_SPECS = {
    "alpha2": {
        "label": "Leader posting-origination advantage",
        "nominal": 1.10,
        "low": 0.55,
        "high": 1.65,
    },
    "beta1_diff": {
        "label": "Leader follower-based diffusion amplification",
        "nominal": 0.80,
        "low": 0.40,
        "high": 1.20,
    },
    "w_l": {
        "label": "Leader exposure weight",
        "nominal": 1.50,
        "low": 1.00,
        "high": 2.00,
    },
    "lambda_L": {
        "label": "Leader attractiveness in network adaptation",
        "nominal": 0.50,
        "low": 0.00,
        "high": 1.00,
    },
}

STUDY_SPECS = {
    "core_robustness": {
        "description": "One-at-a-time perturbations at the 3 percent leader benchmark.",
        "leader_shares": [BENCHMARK_LEADER_SHARE],
        "parameters": ["alpha2", "beta1_diff", "w_l", "lambda_L"],
    },
    "share_gradient": {
        "description": "Share-gradient check for visibility parameters most directly tied to exposure.",
        "leader_shares": SHARE_LEVELS,
        "parameters": ["beta1_diff", "w_l"],
    },
}


def get_profile_spec(profile_name: str) -> dict:
    normalized = str(profile_name).strip().lower()
    if normalized not in PROFILE_SPECS:
        raise ValueError(f"Unsupported sensitivity profile: {profile_name}")
    return dict(PROFILE_SPECS[normalized])


def get_study_spec(study_name: str) -> dict:
    normalized = str(study_name).strip().lower()
    if normalized not in STUDY_SPECS:
        raise ValueError(f"Unsupported sensitivity study: {study_name}")
    return dict(STUDY_SPECS[normalized])


def make_seed_list(seed_count: int, base_seed: int = DEFAULT_SEED) -> list[int]:
    return [int(base_seed + 17 * seed_index) for seed_index in range(seed_count)]


def build_perturbation_catalog(study_name: str) -> pd.DataFrame:
    study = get_study_spec(study_name)
    rows = [
        {
            "perturbation_id": "nominal",
            "varied_parameter": "nominal",
            "parameter_label": "Tuned simple leader-influence reference",
            "parameter_level": "nominal",
            "parameter_value": float("nan"),
            "overrides_json": "{}",
        }
    ]
    for parameter_name in study["parameters"]:
        parameter = PARAMETER_SPECS[parameter_name]
        for level in ("low", "high"):
            value = float(parameter[level])
            rows.append(
                {
                    "perturbation_id": f"{parameter_name}_{level}",
                    "varied_parameter": parameter_name,
                    "parameter_label": parameter["label"],
                    "parameter_level": level,
                    "parameter_value": value,
                    "overrides_json": json.dumps({parameter_name: value}, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)


def parameter_overrides(perturbation_id: str, study_name: str) -> dict:
    if perturbation_id in {"control", "nominal"}:
        return {}
    catalog = build_perturbation_catalog(study_name)
    selected = catalog[catalog["perturbation_id"] == perturbation_id]
    if selected.empty:
        raise ValueError(f"Unsupported perturbation_id for {study_name}: {perturbation_id}")
    return json.loads(selected.iloc[0]["overrides_json"])


def _condition_id(
    study_name: str,
    role: str,
    topology: str,
    leader_share: float,
    leader_mode: str,
    perturbation_id: str,
    rounds: int,
    seed: int,
) -> str:
    share_label = f"{int(round(100 * leader_share)):02d}pct"
    return (
        f"{study_name}_{role}_N{N_AGENTS}_{topology}_{share_label}_"
        f"{leader_mode}_{perturbation_id}_T{rounds}_seed{seed}"
    )


def build_sensitivity_grid(profile_name: str, study_name: str) -> pd.DataFrame:
    profile = get_profile_spec(profile_name)
    study = get_study_spec(study_name)
    seeds = make_seed_list(profile["seed_count"])
    topologies = profile.get("topologies", TOPOLOGIES)
    leader_modes = profile.get("leader_modes", LEADER_MODES)
    catalog = build_perturbation_catalog(study_name)
    selected_ids = profile.get(f"{study_name.split('_')[0]}_perturbation_ids")
    if selected_ids is None and study_name == "core_robustness":
        selected_ids = profile.get("core_perturbation_ids")
    if selected_ids is None and study_name == "share_gradient":
        selected_ids = profile.get("share_perturbation_ids")
    if selected_ids is not None:
        catalog = catalog[catalog["perturbation_id"].isin(selected_ids)].copy()

    rows = []
    rounds = int(profile["T_rounds"])
    for topology in topologies:
        for seed in seeds:
            control_id = _condition_id(study_name, "control", topology, 0.0, "none", "control", rounds, seed)
            rows.append(
                {
                    "profile_name": profile_name,
                    "study_name": study_name,
                    "scenario_name": f"sensitivity_{study_name}",
                    "condition_role": "control",
                    "matched_control_id": control_id,
                    "N": N_AGENTS,
                    "topology": topology,
                    "leader_share": 0.0,
                    "leader_mode": "none",
                    "leader_selection_method": "none",
                    "T_rounds": rounds,
                    "seed": seed,
                    "perturbation_id": "control",
                    "varied_parameter": "none",
                    "parameter_label": "Matched no-leader control",
                    "parameter_level": "control",
                    "parameter_value": float("nan"),
                    "overrides_json": "{}",
                    "condition_id": control_id,
                }
            )

    for _, perturbation in catalog.iterrows():
        for leader_share in study["leader_shares"]:
            for topology in topologies:
                for leader_mode in leader_modes:
                    for seed in seeds:
                        control_id = _condition_id(
                            study_name, "control", topology, 0.0, "none", "control", rounds, seed
                        )
                        rows.append(
                            {
                                "profile_name": profile_name,
                                "study_name": study_name,
                                "scenario_name": f"sensitivity_{study_name}",
                                "condition_role": "leader",
                                "matched_control_id": control_id,
                                "N": N_AGENTS,
                                "topology": topology,
                                "leader_share": float(leader_share),
                                "leader_mode": leader_mode,
                                "leader_selection_method": "top_in_degree",
                                "T_rounds": rounds,
                                "seed": seed,
                                **perturbation.to_dict(),
                                "condition_id": _condition_id(
                                    study_name,
                                    "leader",
                                    topology,
                                    float(leader_share),
                                    leader_mode,
                                    perturbation["perturbation_id"],
                                    rounds,
                                    seed,
                                ),
                            }
                        )
    return pd.DataFrame(rows)
