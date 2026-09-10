"""Integrated opinion-leader and platform reference scenario."""

from opinion_model.scenarios.baseline.assembly import assemble_baseline_components
from opinion_model.scenarios.baseline.config import (
    BaselineConfig,
    OpinionLeaderMechanismConfig,
    PlatformMechanismConfig,
)
from opinion_model.scenarios.baseline.initialization import (
    BaselineInitializer,
    ValidatedBaselineInitializer,
)

__all__ = [
    "BaselineConfig",
    "BaselineInitializer",
    "OpinionLeaderMechanismConfig",
    "PlatformMechanismConfig",
    "ValidatedBaselineInitializer",
    "assemble_baseline_components",
]
