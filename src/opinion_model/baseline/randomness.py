"""Independent, reproducible random streams for baseline mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b

import numpy as np


def _stream_code(name: str) -> int:
    digest = blake2b(name.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


@dataclass(frozen=True)
class RandomStreams:
    base_seed: int

    def _generator(
        self,
        mechanism: str,
        round_index: int = 0,
        entity_id: int = 0,
    ) -> np.random.Generator:
        sequence = np.random.SeedSequence(
            [self.base_seed, _stream_code(mechanism), round_index, entity_id]
        )
        return np.random.default_rng(sequence)

    def initialization(self) -> np.random.Generator:
        return self._generator("initialization")

    def posting(self, round_index: int, agent_id: int) -> np.random.Generator:
        return self._generator("posting", round_index, agent_id)

    def stance(self, round_index: int, agent_id: int) -> np.random.Generator:
        """Preserve the parent notebook's established stance stream."""
        sequence = np.random.SeedSequence([self.base_seed, round_index, agent_id])
        return np.random.default_rng(sequence)

    def selection(self, round_index: int, agent_id: int) -> np.random.Generator:
        return self._generator("selection", round_index, agent_id)

    def network(self, round_index: int) -> np.random.Generator:
        return self._generator("network", round_index)
