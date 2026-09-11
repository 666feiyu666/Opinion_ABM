"""Component assembly for the null scenario."""

from __future__ import annotations

from dataclasses import replace

from opinion_model.scenarios.null.config import NullConfig
from opinion_model.scenarios.null.initialization import (
    NullInitializer,
    ValidatedNullInitializer,
)
from opinion_model.shared import DEFAULT_COMPONENTS, ModelComponents


def assemble_null_components(
    config: NullConfig,
    *,
    initializer: NullInitializer,
) -> ModelComponents:
    """Use only shared ordinary mechanisms and the resolved null initializer."""
    if not isinstance(config, NullConfig):
        raise TypeError("config must be a NullConfig.")
    return replace(
        DEFAULT_COMPONENTS,
        initializer=ValidatedNullInitializer(initializer),
    )


__all__ = ["assemble_null_components"]
