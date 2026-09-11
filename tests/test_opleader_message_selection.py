from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.core import Message, MessageSelection, NetworkState, SelectionContext
from opinion_model.opleader import select_messages


class FailingRng:
    """Fail if deterministic selection attempts to use randomness."""

    def __getattr__(self, name: str):
        raise AssertionError(f"Selection unexpectedly requested rng.{name}.")


class TieBoundMessageSelectionTests(unittest.TestCase):
    LEADER_0 = 0
    LEADER_1 = 1
    ORDINARY_2 = 2
    ORDINARY_3 = 3
    ISOLATED_4 = 4

    def setUp(self) -> None:
        # Reciprocal adjacency encodes fixed regular interpersonal ties:
        # leader 0 -- leader 1, leader 0 -- ordinary 2,
        # and ordinary 2 -- ordinary 3. Agent 4 is isolated.
        self.network = NetworkState(
            {
                self.LEADER_0: (self.LEADER_1, self.ORDINARY_2),
                self.LEADER_1: (self.LEADER_0,),
                self.ORDINARY_2: (self.LEADER_0, self.ORDINARY_3),
                self.ORDINARY_3: (self.ORDINARY_2,),
                self.ISOLATED_4: (),
            }
        )
        self.context = SelectionContext(
            round_index=1,
            capacity=4,
            exclude_self_messages=True,
        )

    @staticmethod
    def message(producer_id: int, stance: int, label: str) -> Message:
        return Message(
            message_id=f"r1:{label}",
            round_index=1,
            producer_id=producer_id,
            stance=stance,
        )

    @staticmethod
    def message_ids(exposures) -> tuple[str, ...]:
        return tuple(exposure.message.message_id for exposure in exposures)

    def select(
        self,
        consumer_id: int,
        messages: tuple[Message, ...],
        *,
        context: SelectionContext | None = None,
        rng=None,
    ):
        return select_messages(
            consumer_id,
            messages,
            self.network,
            context or self.context,
            rng if rng is not None else FailingRng(),
        )

    def test_implements_existing_message_selection_protocol(self) -> None:
        self.assertIsInstance(select_messages, MessageSelection)

    def test_tied_leader_and_ordinary_messages_use_the_same_rule(self) -> None:
        leader = self.message(self.LEADER_0, 1, "leader-0")
        ordinary = self.message(self.ORDINARY_3, -1, "ordinary-3")

        exposures = self.select(self.ORDINARY_2, (ordinary, leader))

        self.assertEqual(
            self.message_ids(exposures),
            ("r1:leader-0", "r1:ordinary-3"),
        )

    def test_leader_to_leader_delivery_is_active_when_tied(self) -> None:
        message = self.message(self.LEADER_1, 1, "leader-1")

        exposures = self.select(self.LEADER_0, (message,))

        self.assertEqual(self.message_ids(exposures), ("r1:leader-1",))

    def test_ordinary_to_ordinary_delivery_is_active_when_tied(self) -> None:
        message = self.message(self.ORDINARY_3, -1, "ordinary-3")

        exposures = self.select(self.ORDINARY_2, (message,))

        self.assertEqual(self.message_ids(exposures), ("r1:ordinary-3",))

    def test_untied_messages_are_not_delivered(self) -> None:
        untied = self.message(self.ORDINARY_3, 1, "ordinary-3")

        exposures = self.select(self.LEADER_0, (untied,))

        self.assertEqual(exposures, ())

    def test_isolated_agent_receives_no_messages(self) -> None:
        messages = (
            self.message(self.LEADER_0, 1, "leader-0"),
            self.message(self.ORDINARY_2, -1, "ordinary-2"),
        )

        exposures = self.select(self.ISOLATED_4, messages)

        self.assertEqual(exposures, ())

    def test_every_eligible_message_is_delivered_exactly_once(self) -> None:
        messages = (
            self.message(self.LEADER_1, -1, "leader-1"),
            self.message(self.ORDINARY_2, 1, "ordinary-2"),
        )

        exposures = self.select(self.LEADER_0, messages)

        self.assertEqual(
            self.message_ids(exposures),
            ("r1:leader-1", "r1:ordinary-2"),
        )

    def test_selection_is_stance_independent_and_preserves_message_identity(self) -> None:
        support = self.message(self.LEADER_0, 1, "leader")
        oppose = self.message(self.LEADER_0, -1, "leader")

        support_exposure = self.select(self.ORDINARY_2, (support,))[0]
        oppose_exposure = self.select(self.ORDINARY_2, (oppose,))[0]

        self.assertIs(support_exposure.message, support)
        self.assertIs(oppose_exposure.message, oppose)
        self.assertEqual(support_exposure.message.producer_id, self.LEADER_0)
        self.assertEqual(oppose_exposure.message.producer_id, self.LEADER_0)
        self.assertEqual(support_exposure.message.stance, 1)
        self.assertEqual(oppose_exposure.message.stance, -1)

    def test_message_pool_order_and_seed_do_not_change_selection(self) -> None:
        messages = (
            self.message(self.ORDINARY_2, -1, "ordinary-2"),
            self.message(self.LEADER_1, 1, "leader-1"),
        )

        first = self.select(
            self.LEADER_0,
            messages,
            rng=np.random.default_rng(1),
        )
        second = self.select(
            self.LEADER_0,
            tuple(reversed(messages)),
            rng=np.random.default_rng(999),
        )

        self.assertEqual(first, second)

    def test_deterministic_selection_never_uses_rng(self) -> None:
        message = self.message(self.LEADER_0, 1, "leader-0")

        exposures = self.select(self.ORDINARY_2, (message,), rng=FailingRng())

        self.assertEqual(self.message_ids(exposures), ("r1:leader-0",))

    def test_capacity_equal_to_eligible_count_is_accepted(self) -> None:
        messages = (
            self.message(self.LEADER_1, 1, "leader-1"),
            self.message(self.ORDINARY_2, -1, "ordinary-2"),
        )
        context = SelectionContext(1, 2, True)

        exposures = self.select(self.LEADER_0, messages, context=context)

        self.assertEqual(len(exposures), 2)

    def test_binding_capacity_is_rejected(self) -> None:
        messages = (
            self.message(self.LEADER_1, 1, "leader-1"),
            self.message(self.ORDINARY_2, -1, "ordinary-2"),
        )
        context = SelectionContext(1, 1, True)

        with self.assertRaisesRegex(ValueError, "attention competition is outside"):
            self.select(self.LEADER_0, messages, context=context)

    def test_unknown_consumer_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not registered in the network"):
            self.select(99, ())

    def test_unknown_producer_is_rejected(self) -> None:
        unknown = self.message(99, 1, "unknown")

        with self.assertRaisesRegex(
            ValueError,
            "Producer 99 is not registered in the network",
        ):
            self.select(self.LEADER_0, (unknown,))

    def test_message_round_must_match_selection_round(self) -> None:
        stale = Message(
            message_id="r2:leader-0",
            round_index=2,
            producer_id=self.LEADER_0,
            stance=1,
        )

        with self.assertRaisesRegex(ValueError, "not selection round 1"):
            self.select(self.ORDINARY_2, (stale,))

    def test_self_message_is_not_delivered(self) -> None:
        own_message = self.message(self.LEADER_0, 1, "leader-0")

        exposures = self.select(self.LEADER_0, (own_message,))

        self.assertEqual(exposures, ())

    def test_selection_does_not_mutate_inputs(self) -> None:
        message = self.message(self.LEADER_0, 1, "leader-0")
        pool = (message,)
        before_network = self.network

        self.select(self.ORDINARY_2, pool)

        self.assertEqual(pool, (message,))
        self.assertIs(self.network, before_network)


if __name__ == "__main__":
    unittest.main()
