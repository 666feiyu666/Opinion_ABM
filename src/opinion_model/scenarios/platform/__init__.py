"""Matched platform-mediated scenario without opinion-leader mechanisms."""

from opinion_model.scenarios.platform.assembly import assemble_platform_components
from opinion_model.scenarios.platform.config import (
    PlatformConfig,
    PlatformExperimentConfig,
    PlatformInitializationConfig,
    PlatformMechanismConfig,
    load_platform_experiment_config,
)
from opinion_model.scenarios.platform.experiment import (
    PlatformRunResult,
    final_condition_metrics,
    run_platform_condition,
    run_platform_experiment,
    summarize_final_conditions,
)
from opinion_model.scenarios.platform.initialization import (
    FixedPlatformInitializer,
    PlatformInitialization,
    initialize_platform,
)
from opinion_model.scenarios.platform.observation import (
    platform_diagnostic_frames,
    platform_network_decision_frame,
    platform_round_metrics,
    platform_selection_decision_frame,
    platform_transition_ledger,
)


__all__ = [
    "FixedPlatformInitializer",
    "PlatformConfig",
    "PlatformExperimentConfig",
    "PlatformInitialization",
    "PlatformInitializationConfig",
    "PlatformMechanismConfig",
    "PlatformRunResult",
    "assemble_platform_components",
    "final_condition_metrics",
    "initialize_platform",
    "load_platform_experiment_config",
    "platform_diagnostic_frames",
    "platform_network_decision_frame",
    "platform_round_metrics",
    "platform_selection_decision_frame",
    "platform_transition_ledger",
    "run_platform_condition",
    "run_platform_experiment",
    "summarize_final_conditions",
]
