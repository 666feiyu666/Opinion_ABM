"""Shared state, event, and context objects for opinion-model simulations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return value


@dataclass(frozen=True)
class BetaBelief:
    """Private uncertainty over support for the positive proposition."""

    a: float
    b: float

    def __post_init__(self) -> None:
        if not (isfinite(self.a) and isfinite(self.b)):
            raise ValueError("Beta shape parameters must be finite.")
        if self.a <= 0.0 or self.b <= 0.0:
            raise ValueError("Beta shape parameters must be positive.")

    @property
    def concentration(self) -> float:
        return self.a + self.b

    @property
    def mean(self) -> float:
        return self.a / self.concentration

    @property
    def signed_mean(self) -> float:
        return 2.0 * self.mean - 1.0


@dataclass(frozen=True)
class AgentState:
    """Minimal shared state of one agent."""

    belief: BetaBelief


@dataclass(frozen=True)
class Message:
    """One observable binary message."""

    message_id: str
    round_index: int
    producer_id: int
    stance: int

    def __post_init__(self) -> None:
        if not self.message_id:
            raise ValueError("message_id must be non-empty.")
        _nonnegative_integer(self.round_index, "round_index")
        _nonnegative_integer(self.producer_id, "producer_id")
        if self.round_index == 0:
            raise ValueError("Messages are produced only in rounds 1 and later.")
        if self.stance not in (-1, 1):
            raise ValueError("Message stance must be -1 or +1.")


@dataclass(frozen=True)
class ProductionOutcome:
    """Result of giving one agent one production opportunity."""

    round_index: int
    agent_id: int
    did_post: bool
    post_probability: float
    support_probability: float
    message: Message | None

    def __post_init__(self) -> None:
        _nonnegative_integer(self.round_index, "round_index")
        _nonnegative_integer(self.agent_id, "agent_id")
        if self.round_index == 0:
            raise ValueError("Production occurs only in rounds 1 and later.")
        if not isinstance(self.did_post, bool):
            raise ValueError("did_post must be Boolean.")
        _probability(self.post_probability, "post_probability")
        _probability(self.support_probability, "support_probability")
        if self.did_post != (self.message is not None):
            raise ValueError("did_post and message presence must agree.")
        if self.message is not None:
            if self.message.round_index != self.round_index:
                raise ValueError("Outcome and message rounds must agree.")
            if self.message.producer_id != self.agent_id:
                raise ValueError("Outcome agent and message producer must agree.")


@dataclass(frozen=True)
class Exposure:
    """Delivery of one produced message to one consumer."""

    round_index: int
    consumer_id: int
    message: Message

    def __post_init__(self) -> None:
        _nonnegative_integer(self.round_index, "round_index")
        _nonnegative_integer(self.consumer_id, "consumer_id")
        if self.message.round_index != self.round_index:
            raise ValueError("Exposure and message rounds must agree.")


@dataclass(frozen=True)
class MessageEvidence:
    """Explicit aggregate passed from exposure to opinion updating."""

    n_support: int
    n_oppose: int
    weighted_support: float
    weighted_oppose: float

    def __post_init__(self) -> None:
        _nonnegative_integer(self.n_support, "n_support")
        _nonnegative_integer(self.n_oppose, "n_oppose")
        if not (isfinite(self.weighted_support) and self.weighted_support >= 0.0):
            raise ValueError("weighted_support must be finite and non-negative.")
        if not (isfinite(self.weighted_oppose) and self.weighted_oppose >= 0.0):
            raise ValueError("weighted_oppose must be finite and non-negative.")

    @property
    def total_messages(self) -> int:
        return self.n_support + self.n_oppose

    @property
    def total_weight(self) -> float:
        return self.weighted_support + self.weighted_oppose


@dataclass(frozen=True)
class NetworkState:
    """Directed source eligibility for each potential consumer."""

    neighbors_by_agent: Mapping[int, tuple[int, ...]]

    def __post_init__(self) -> None:
        normalized: dict[int, tuple[int, ...]] = {}
        for agent_id, neighbors in self.neighbors_by_agent.items():
            _nonnegative_integer(agent_id, "agent_id")
            normalized_neighbors = tuple(sorted(set(neighbors)))
            for neighbor_id in normalized_neighbors:
                _nonnegative_integer(neighbor_id, "neighbor_id")
            if agent_id in normalized_neighbors:
                raise ValueError("Self-links are not allowed in NetworkState.")
            normalized[agent_id] = normalized_neighbors
        object.__setattr__(
            self,
            "neighbors_by_agent",
            MappingProxyType(dict(sorted(normalized.items()))),
        )

    def eligible_producers(self, consumer_id: int) -> tuple[int, ...]:
        return self.neighbors_by_agent.get(consumer_id, ())


@dataclass(frozen=True)
class WorldState:
    """Immutable state snapshot read by every process in one round."""

    round_index: int
    agents: Mapping[int, AgentState]
    network: NetworkState

    def __post_init__(self) -> None:
        _nonnegative_integer(self.round_index, "round_index")
        normalized = dict(sorted(self.agents.items()))
        for agent_id, state in normalized.items():
            _nonnegative_integer(agent_id, "agent_id")
            if not isinstance(state, AgentState):
                raise ValueError("Every agents value must be an AgentState.")
        if set(normalized) != set(self.network.neighbors_by_agent):
            raise ValueError("Agent and network node IDs must agree.")
        object.__setattr__(self, "agents", MappingProxyType(normalized))


@dataclass(frozen=True)
class ProductionContext:
    round_index: int
    post_probability: float

    def __post_init__(self) -> None:
        _nonnegative_integer(self.round_index, "round_index")
        _probability(self.post_probability, "post_probability")


@dataclass(frozen=True)
class SelectionContext:
    round_index: int
    capacity: int
    exclude_self_messages: bool

    def __post_init__(self) -> None:
        _nonnegative_integer(self.round_index, "round_index")
        _nonnegative_integer(self.capacity, "capacity")
        if not isinstance(self.exclude_self_messages, bool):
            raise ValueError("exclude_self_messages must be Boolean.")


@dataclass(frozen=True)
class AggregationContext:
    evidence_weight: float

    def __post_init__(self) -> None:
        if not isfinite(self.evidence_weight) or self.evidence_weight < 0.0:
            raise ValueError("evidence_weight must be finite and non-negative.")


@dataclass(frozen=True)
class NetworkUpdateContext:
    round_index: int

    def __post_init__(self) -> None:
        _nonnegative_integer(self.round_index, "round_index")


@dataclass(frozen=True)
class RoundEvents:
    """Complete events and aggregates produced before synchronous commit."""

    production_outcomes: tuple[ProductionOutcome, ...]
    exposures: tuple[Exposure, ...]
    evidence_by_agent: Mapping[int, MessageEvidence]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_by_agent",
            MappingProxyType(dict(sorted(self.evidence_by_agent.items()))),
        )
