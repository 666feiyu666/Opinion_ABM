from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opinion_model.core import Message, MessageSelection, NetworkState, SelectionContext
from opinion_model.platform import PlatformMessageSelection


class ControlledRng:
    def __init__(self, random_values=(), choice_indices=()):
        self._random_values = iter(random_values)
        self.choice_indices = tuple(choice_indices)
        self.random_calls = 0
        self.choice_calls = 0

    def random(self):
        self.random_calls += 1
        return next(self._random_values)

    def choice(self, population_size, *, size, replace):
        self.choice_calls += 1
        if replace:
            raise AssertionError("Capacity sampling must be without replacement.")
        if len(self.choice_indices) != size:
            raise AssertionError("The controlled choice does not match requested size.")
        if any(index >= population_size for index in self.choice_indices):
            raise AssertionError("Controlled choice index is outside the candidate pool.")
        return np.asarray(self.choice_indices)


class PlatformMessageSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = NetworkState({0: (1,), 1: (), 2: (), 3: ()})
        self.context = SelectionContext(1, 3, True)

    @staticmethod
    def message(producer_id: int, stance: int = 1) -> Message:
        return Message(
            message_id=f"r1:a{producer_id}",
            round_index=1,
            producer_id=producer_id,
            stance=stance,
        )

    @staticmethod
    def producer_ids(exposures) -> tuple[int, ...]:
        return tuple(exposure.message.producer_id for exposure in exposures)

    def test_implements_message_selection_protocol(self) -> None:
        self.assertIsInstance(PlatformMessageSelection(0.5), MessageSelection)

    def test_tied_messages_are_available_without_a_bernoulli_draw(self) -> None:
        rng = ControlledRng()
        exposures = PlatformMessageSelection(0.0)(
            0,
            (self.message(1),),
            self.network,
            self.context,
            rng,
        )
        self.assertEqual(self.producer_ids(exposures), (1,))
        self.assertEqual(rng.random_calls, 0)

    def test_zero_and_one_are_exact_out_of_network_boundaries(self) -> None:
        message = self.message(2)
        absent = PlatformMessageSelection(0.0)(
            0,
            (message,),
            self.network,
            self.context,
            ControlledRng((0.0,)),
        )
        present = PlatformMessageSelection(1.0)(
            0,
            (message,),
            self.network,
            self.context,
            ControlledRng((0.999999,)),
        )
        self.assertEqual(absent, ())
        self.assertEqual(self.producer_ids(present), (2,))

    def test_each_out_of_network_message_receives_an_independent_draw(self) -> None:
        rng = ControlledRng((0.2, 0.8))
        exposures = PlatformMessageSelection(0.5)(
            0,
            (self.message(3), self.message(2)),
            self.network,
            self.context,
            rng,
        )
        self.assertEqual(self.producer_ids(exposures), (2,))
        self.assertEqual(rng.random_calls, 2)

    def test_tied_and_out_of_network_candidates_share_one_capacity_pool(self) -> None:
        rng = ControlledRng((0.0, 0.0), choice_indices=(1, 2))
        exposures = PlatformMessageSelection(1.0)(
            0,
            (self.message(3), self.message(1), self.message(2)),
            self.network,
            SelectionContext(1, 2, True),
            rng,
        )
        self.assertEqual(self.producer_ids(exposures), (2, 3))
        self.assertEqual(rng.choice_calls, 1)

    def test_pool_order_does_not_change_a_seeded_result(self) -> None:
        messages = (self.message(3), self.message(1), self.message(2))
        rule = PlatformMessageSelection(0.7)
        first = rule(
            0,
            messages,
            self.network,
            SelectionContext(1, 2, True),
            np.random.default_rng(17),
        )
        second = rule(
            0,
            tuple(reversed(messages)),
            self.network,
            SelectionContext(1, 2, True),
            np.random.default_rng(17),
        )
        self.assertEqual(first, second)

    def test_self_messages_are_excluded_before_availability_sampling(self) -> None:
        rng = ControlledRng()
        exposures = PlatformMessageSelection(1.0)(
            0,
            (self.message(0),),
            self.network,
            self.context,
            rng,
        )
        self.assertEqual(exposures, ())
        self.assertEqual(rng.random_calls, 0)

    def test_stance_does_not_change_selection(self) -> None:
        support = PlatformMessageSelection(1.0)(
            0,
            (self.message(2, 1),),
            self.network,
            self.context,
            ControlledRng((0.3,)),
        )
        oppose = PlatformMessageSelection(1.0)(
            0,
            (self.message(2, -1),),
            self.network,
            self.context,
            ControlledRng((0.3,)),
        )
        self.assertEqual(len(support), len(oppose))
        self.assertEqual(support[0].message.producer_id, oppose[0].message.producer_id)

    def test_invalid_probability_is_rejected(self) -> None:
        for value in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PlatformMessageSelection(value)

    def test_unknown_consumer_and_producer_are_rejected(self) -> None:
        rule = PlatformMessageSelection(0.5)
        with self.assertRaisesRegex(ValueError, "Consumer 99"):
            rule(99, (), self.network, self.context, ControlledRng())
        with self.assertRaisesRegex(ValueError, "Producer 99"):
            rule(
                0,
                (
                    Message(
                        message_id="r1:unknown",
                        round_index=1,
                        producer_id=99,
                        stance=1,
                    ),
                ),
                self.network,
                self.context,
                ControlledRng(),
            )

    def test_message_round_must_match_selection_round(self) -> None:
        stale = Message("r2:a2", 2, 2, 1)
        with self.assertRaisesRegex(ValueError, "not selection round 1"):
            PlatformMessageSelection(0.5)(
                0,
                (stale,),
                self.network,
                self.context,
                ControlledRng(),
            )


if __name__ == "__main__":
    unittest.main()
