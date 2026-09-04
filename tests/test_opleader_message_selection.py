from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.core import Message, MessageSelection, NetworkState, SelectionContext
from opinion_model.opleader import (
    OpinionLeaderMessageSelection,
    OriginatorKind,
    RecipientKind,
)


class CountingRng:
    def __init__(self, *values: float) -> None:
        self.values = list(values)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        if not self.values:
            raise AssertionError("Unexpected random draw.")
        return self.values.pop(0)


class OpinionLeaderMessageSelectionTests(unittest.TestCase):
    LEADER_0 = 0
    LEADER_1 = 1
    ORDINARY_2 = 2
    ORDINARY_3 = 3
    ORDINARY_4 = 4
    PRESS_ID = 10

    def setUp(self) -> None:
        self.network = NetworkState(
            {
                self.LEADER_0: (),
                self.LEADER_1: (),
                self.ORDINARY_2: (self.LEADER_0,),
                self.ORDINARY_3: (self.LEADER_0, self.LEADER_1),
                self.ORDINARY_4: (),
            }
        )
        self.selector = OpinionLeaderMessageSelection(
            originator_kind_by_id={
                self.LEADER_0: OriginatorKind.LEADER,
                self.LEADER_1: OriginatorKind.LEADER,
                self.PRESS_ID: OriginatorKind.PRESS,
            },
            recipient_kind_by_id={
                self.LEADER_0: RecipientKind.LEADER,
                self.LEADER_1: RecipientKind.LEADER,
                self.ORDINARY_2: RecipientKind.ORDINARY,
                self.ORDINARY_3: RecipientKind.ORDINARY,
                self.ORDINARY_4: RecipientKind.ORDINARY,
            },
            press_delivery_probability_by_recipient_kind={
                RecipientKind.LEADER: 0.8,
                RecipientKind.ORDINARY: 0.2,
            },
        )
        self.context = SelectionContext(
            round_index=1,
            capacity=10,
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

    def message_ids(self, exposures) -> tuple[str, ...]:
        return tuple(exposure.message.message_id for exposure in exposures)

    def test_implements_existing_message_selection_protocol(self) -> None:
        self.assertIsInstance(self.selector, MessageSelection)

    def test_press_probability_depends_on_recipient_role(self) -> None:
        press = self.message(self.PRESS_ID, 1, "press")

        leader_exposures = self.selector(
            self.LEADER_0,
            (press,),
            self.network,
            self.context,
            CountingRng(0.5),
        )
        ordinary_exposures = self.selector(
            self.ORDINARY_2,
            (press,),
            self.network,
            self.context,
            CountingRng(0.5),
        )

        self.assertEqual(self.message_ids(leader_exposures), ("r1:press",))
        self.assertEqual(ordinary_exposures, ())

    def test_press_zero_and_unit_probability_boundaries(self) -> None:
        press = self.message(self.PRESS_ID, -1, "press")
        selector = OpinionLeaderMessageSelection(
            originator_kind_by_id=self.selector.originator_kind_by_id,
            recipient_kind_by_id=self.selector.recipient_kind_by_id,
            press_delivery_probability_by_recipient_kind={
                RecipientKind.LEADER: 0.0,
                RecipientKind.ORDINARY: 1.0,
            },
        )
        leader_rng = CountingRng(0.0)
        ordinary_rng = CountingRng(0.999)

        self.assertEqual(
            selector(
                self.LEADER_0,
                (press,),
                self.network,
                self.context,
                leader_rng,
            ),
            (),
        )
        ordinary_exposures = selector(
            self.ORDINARY_2,
            (press,),
            self.network,
            self.context,
            ordinary_rng,
        )
        self.assertEqual(self.message_ids(ordinary_exposures), ("r1:press",))
        self.assertEqual(leader_rng.calls, 1)
        self.assertEqual(ordinary_rng.calls, 1)

    def test_press_bypasses_network_ties(self) -> None:
        press = self.message(self.PRESS_ID, 1, "press")
        empty_tie_recipient = self.ORDINARY_4
        exposures = self.selector(
            empty_tie_recipient,
            (press,),
            self.network,
            self.context,
            CountingRng(0.1),
        )

        self.assertEqual(self.message_ids(exposures), ("r1:press",))
        self.assertNotIn(
            self.PRESS_ID,
            self.network.eligible_producers(empty_tie_recipient),
        )

    def test_leader_message_requires_incoming_tie_to_ordinary_recipient(self) -> None:
        leader_0 = self.message(self.LEADER_0, 1, "leader-0")
        leader_1 = self.message(self.LEADER_1, -1, "leader-1")

        tied = self.selector(
            self.ORDINARY_2,
            (leader_0, leader_1),
            self.network,
            self.context,
            CountingRng(),
        )
        untied = self.selector(
            self.ORDINARY_4,
            (leader_0, leader_1),
            self.network,
            self.context,
            CountingRng(),
        )

        self.assertEqual(self.message_ids(tied), ("r1:leader-0",))
        self.assertEqual(untied, ())

    def test_ordinary_recipient_can_receive_multiple_tied_leaders(self) -> None:
        messages = (
            self.message(self.LEADER_1, -1, "leader-1"),
            self.message(self.LEADER_0, 1, "leader-0"),
        )
        exposures = self.selector(
            self.ORDINARY_3,
            messages,
            self.network,
            self.context,
            CountingRng(),
        )

        self.assertEqual(
            self.message_ids(exposures),
            ("r1:leader-0", "r1:leader-1"),
        )

    def test_network_direction_is_consumer_to_eligible_producer(self) -> None:
        reverse_only_network = NetworkState(
            {
                self.LEADER_0: (self.ORDINARY_2,),
                self.LEADER_1: (),
                self.ORDINARY_2: (),
                self.ORDINARY_3: (),
                self.ORDINARY_4: (),
            }
        )
        message = self.message(self.LEADER_0, 1, "leader-0")

        exposures = self.selector(
            self.ORDINARY_2,
            (message,),
            reverse_only_network,
            self.context,
            CountingRng(),
        )

        self.assertEqual(exposures, ())

    def test_leader_to_leader_delivery_is_inactive_even_with_tie(self) -> None:
        network_with_leader_tie = NetworkState(
            {
                self.LEADER_0: (self.LEADER_1,),
                self.LEADER_1: (),
                self.ORDINARY_2: (self.LEADER_0,),
                self.ORDINARY_3: (),
                self.ORDINARY_4: (),
            }
        )
        message = self.message(self.LEADER_1, 1, "leader-1")

        exposures = self.selector(
            self.LEADER_0,
            (message,),
            network_with_leader_tie,
            self.context,
            CountingRng(),
        )

        self.assertEqual(exposures, ())

    def test_selection_is_stance_blind_and_preserves_message_identity(self) -> None:
        positive = self.message(self.LEADER_0, 1, "leader")
        negative = self.message(self.LEADER_0, -1, "leader")

        positive_exposures = self.selector(
            self.ORDINARY_2,
            (positive,),
            self.network,
            self.context,
            CountingRng(),
        )
        negative_exposures = self.selector(
            self.ORDINARY_2,
            (negative,),
            self.network,
            self.context,
            CountingRng(),
        )

        self.assertEqual(
            self.message_ids(positive_exposures),
            self.message_ids(negative_exposures),
        )
        self.assertIs(positive_exposures[0].message, positive)
        self.assertIs(negative_exposures[0].message, negative)

    def test_message_pool_order_does_not_change_seeded_result(self) -> None:
        messages = (
            self.message(self.PRESS_ID, 1, "press-b"),
            self.message(self.LEADER_0, -1, "leader"),
            self.message(self.PRESS_ID, -1, "press-a"),
        )

        first = self.selector(
            self.ORDINARY_2,
            messages,
            self.network,
            self.context,
            np.random.default_rng(20260904),
        )
        second = self.selector(
            self.ORDINARY_2,
            tuple(reversed(messages)),
            self.network,
            self.context,
            np.random.default_rng(20260904),
        )

        self.assertEqual(first, second)

    def test_fixed_seed_is_reproducible(self) -> None:
        press = self.message(self.PRESS_ID, 1, "press")

        def run(seed: int) -> tuple[bool, ...]:
            rng = np.random.default_rng(seed)
            return tuple(
                bool(
                    self.selector(
                        self.ORDINARY_2,
                        (press,),
                        self.network,
                        self.context,
                        rng,
                    )
                )
                for _ in range(30)
            )

        self.assertEqual(run(7), run(7))

    def test_selection_does_not_mutate_inputs(self) -> None:
        press = self.message(self.PRESS_ID, 1, "press")
        pool = (press,)
        before_network = self.network

        self.selector(
            self.ORDINARY_2,
            pool,
            self.network,
            self.context,
            CountingRng(0.1),
        )

        self.assertEqual(pool, (press,))
        self.assertIs(self.network, before_network)

    def test_binding_capacity_is_rejected_instead_of_truncating(self) -> None:
        messages = (
            self.message(self.LEADER_0, 1, "leader-0"),
            self.message(self.LEADER_1, -1, "leader-1"),
        )
        context = SelectionContext(1, 1, True)

        with self.assertRaisesRegex(ValueError, "attention competition is omitted"):
            self.selector(
                self.ORDINARY_3,
                messages,
                self.network,
                context,
                CountingRng(),
            )

    def test_unknown_consumer_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "No recipient kind registered"):
            self.selector(
                99,
                (),
                self.network,
                self.context,
                CountingRng(),
            )

    def test_consumer_missing_from_network_is_rejected(self) -> None:
        incomplete_network = NetworkState({self.LEADER_0: (), self.LEADER_1: ()})
        with self.assertRaisesRegex(ValueError, "not registered in the network"):
            self.selector(
                self.ORDINARY_2,
                (),
                incomplete_network,
                self.context,
                CountingRng(),
            )

    def test_unknown_message_originator_is_rejected(self) -> None:
        unknown = self.message(99, 1, "unknown")
        with self.assertRaisesRegex(ValueError, "No originator kind registered"):
            self.selector(
                self.ORDINARY_2,
                (unknown,),
                self.network,
                self.context,
                CountingRng(),
            )

    def test_message_round_must_match_selection_round(self) -> None:
        stale = Message(
            message_id="r2:press",
            round_index=2,
            producer_id=self.PRESS_ID,
            stance=1,
        )
        with self.assertRaisesRegex(ValueError, "not selection round 1"):
            self.selector(
                self.ORDINARY_2,
                (stale,),
                self.network,
                self.context,
                CountingRng(),
            )

    def test_missing_press_probability_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Missing press delivery probabilities for: ordinary",
        ):
            OpinionLeaderMessageSelection(
                originator_kind_by_id={self.PRESS_ID: OriginatorKind.PRESS},
                recipient_kind_by_id={self.ORDINARY_2: RecipientKind.ORDINARY},
                press_delivery_probability_by_recipient_kind={
                    RecipientKind.LEADER: 0.5,
                },
            )

    def test_invalid_press_probability_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite values in"):
            OpinionLeaderMessageSelection(
                originator_kind_by_id={self.PRESS_ID: OriginatorKind.PRESS},
                recipient_kind_by_id={self.ORDINARY_2: RecipientKind.ORDINARY},
                press_delivery_probability_by_recipient_kind={
                    RecipientKind.LEADER: 0.5,
                    RecipientKind.ORDINARY: float("nan"),
                },
            )

    def test_press_cannot_be_registered_as_adaptive_recipient(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot also be an adaptive recipient"):
            OpinionLeaderMessageSelection(
                originator_kind_by_id={self.PRESS_ID: OriginatorKind.PRESS},
                recipient_kind_by_id={self.PRESS_ID: RecipientKind.ORDINARY},
                press_delivery_probability_by_recipient_kind={
                    RecipientKind.LEADER: 0.5,
                    RecipientKind.ORDINARY: 0.5,
                },
            )


if __name__ == "__main__":
    unittest.main()
