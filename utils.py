from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_random_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clip_opinion(x):
    return float(np.clip(x, -1.0, 1.0))


def sign_opinion(x):
    # Legacy helper retained for compatibility with external notebooks/scripts.
    return 1 if x >= 0 else -1


def sample_stance_from_opinion(
    opinion: float,
    confidence: float,
    eta_expression: float = 1.0,
    rng: np.random.Generator | None = None,
) -> int:
    opinion_clipped = clip_opinion(opinion)
    probability_positive = (opinion_clipped + 1.0) / 2.0
    probability_positive = float(np.clip(probability_positive, 0.0, 1.0))

    concentration = max(0.0, eta_expression * float(confidence))
    alpha = 1.0 + concentration * probability_positive
    beta = 1.0 + concentration * (1.0 - probability_positive)
    stance_probability = alpha / (alpha + beta)

    rng_local = rng if rng is not None else np.random.default_rng()
    return 1 if rng_local.random() < stance_probability else -1


def tanh_mapping(o, kappa):
    return float(np.tanh(kappa * o))


def as_history_frame(round_records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(round_records)
