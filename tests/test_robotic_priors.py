import math
import unittest

import torch

from hypercrl.srl.robotic_priors import SlownessPrior, VariabilityPrior, RepeatabilityPrior, ProportionalityPrior, \
    CausalityPrior, ReferencePointPrior


class TestRoboticPriors(unittest.TestCase):
    """
    Unittests for the robotic priors
    """

    def setUp(self) -> None:
        self.dimension = 128
        self.batch_size = 8

    def test_slowness_prior(self) -> None:
        """
        Tests that the slowness prior is calculated as expected.
        """
        state = torch.Tensor([[1] * self.dimension] * self.batch_size)
        next_state = torch.Tensor([[0] * self.dimension] * self.batch_size)
        slowness_prior = SlownessPrior()

        value = slowness_prior(state, next_state)
        self.assertAlmostEqual(value.item(), self.dimension, places=4)

        value = slowness_prior(state, state)
        self.assertEqual(value.item(), 0)

    def test_variability_prior(self) -> None:
        """
        Tests that the variability prior is calculated as expected.
        """
        state = torch.Tensor([[1] * self.dimension] * self.batch_size)
        next_state = torch.Tensor([[0] * self.dimension] * self.batch_size)
        variability_prior = VariabilityPrior()

        value = variability_prior(state, next_state)
        self.assertAlmostEqual(value.item(), math.exp(-(self.dimension ** 0.5)), places=4)

        value = variability_prior(state, state)
        self.assertEqual(value.item(), 1)

    def test_proportionality_prior(self) -> None:
        """
        Tests that the proportionality prior is calculated as expected.
        """
        state = torch.Tensor([[1] * self.dimension] * self.batch_size)
        next_state = torch.Tensor([[0] * self.dimension] * self.batch_size)
        proportionality_prior = ProportionalityPrior()

        value = proportionality_prior(state, next_state, state, state)
        self.assertAlmostEqual(value.item(), self.dimension, places=4)

        value = proportionality_prior(state, next_state, state, next_state)
        self.assertEqual(value.item(), 0)

    def test_repeatability_prior(self) -> None:
        """
        Tests that the repeatability prior is calculated as expected.
        """
        state = torch.Tensor([[1] * self.dimension] * self.batch_size)
        next_state = torch.Tensor([[0] * self.dimension] * self.batch_size)
        repeatability_prior = RepeatabilityPrior()

        value = repeatability_prior(state, next_state, state, state)
        self.assertAlmostEqual(value.item(), self.dimension, places=4)

        value = repeatability_prior(state, next_state, state, next_state)
        self.assertEqual(value.item(), 0)

    def test_causality_prior(self) -> None:
        """
        Tests that the causality prior is calculated as expected.
        """
        state = torch.Tensor([[1] * self.dimension] * self.batch_size)
        next_state = torch.Tensor([[0] * self.dimension] * self.batch_size)
        causality_prior = CausalityPrior()

        value = causality_prior(state, next_state)
        self.assertAlmostEqual(value.item(), 0, places=4)

        value = causality_prior(state, state)
        self.assertEqual(value.item(), 1)

    def test_reference_point_prior(self) -> None:
        """
        Tests that the reference point prior is calculated as expected.
        """
        state = torch.Tensor([[1] * self.dimension] * self.batch_size)
        next_state = torch.Tensor([[0] * self.dimension] * self.batch_size)
        reference_point_prior = ReferencePointPrior()

        value = reference_point_prior(state, next_state)
        self.assertAlmostEqual(value.item(), self.dimension, places=4)

        value = reference_point_prior(state, state)
        self.assertEqual(value.item(), 0)


if __name__ == '__main__':
    unittest.main()
