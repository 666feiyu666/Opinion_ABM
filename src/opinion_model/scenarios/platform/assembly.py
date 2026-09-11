"""Component assembly for the platform-only scenario."""

from __future__ import annotations

from dataclasses import replace

from opinion_model.scenarios.platform.config import PlatformConfig
from opinion_model.scenarios.platform.initialization import (
    PlatformInitializer,
    ValidatedPlatformInitializer,
)
from opinion_model.shared import DEFAULT_COMPONENTS, ModelComponents


def assemble_platform_components(
    config: PlatformConfig,
    *,
    initializer: PlatformInitializer,
) -> ModelComponents:
    """Compose platform mechanisms with ordinary shared behavior."""
    if not isinstance(config, PlatformConfig):
        raise TypeError("config must be a PlatformConfig.")
    return replace(
        DEFAULT_COMPONENTS,
        initializer=ValidatedPlatformInitializer(initializer),
        message_selection=config.platform.build_message_selection(),
        network_update=config.platform.build_network_update(),
    )


__all__ = ["assemble_platform_components"]
