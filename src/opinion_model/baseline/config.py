"""Configuration values for the coupled null baseline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SimulationConfig:
    """Exogenous settings; behavior is supplied separately as components."""

    agent_count: int = 11
    rounds: int = 10
    seed: int = 20260903
    initial_mean: float = 0.5
    initial_concentration: float = 4.0
    post_probability: float = 1.0
    evidence_weight: float = 0.1
    consumption_capacity: int = 10
    exclude_self_messages: bool = True

    def __post_init__(self) -> None:
        for name in ("agent_count", "rounds", "seed", "consumption_capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.agent_count < 2:
            raise ValueError("agent_count must be at least two.")
        if not 0.0 < self.initial_mean < 1.0:
            raise ValueError("initial_mean must lie strictly between 0 and 1.")
        if not isfinite(self.initial_concentration) or self.initial_concentration <= 0.0:
            raise ValueError("initial_concentration must be finite and positive.")
        if not isfinite(self.post_probability) or not 0.0 <= self.post_probability <= 1.0:
            raise ValueError("post_probability must lie in [0, 1].")
        if not isfinite(self.evidence_weight) or self.evidence_weight < 0.0:
            raise ValueError("evidence_weight must be finite and non-negative.")
        if not isinstance(self.exclude_self_messages, bool):
            raise ValueError("exclude_self_messages must be Boolean.")
        eligible_limit = self.agent_count - 1 if self.exclude_self_messages else self.agent_count
        if self.consumption_capacity > eligible_limit:
            raise ValueError("consumption_capacity exceeds the maximum eligible source count.")
