import unittest
import numpy as np

from hypercrl.srl.utils import sample_by_size, probabilities_by_size, remove_and_move


class TestBase(unittest.TestCase):
    """
    Unittests for the utils
    """

    def setUp(self) -> None:
        np.random.seed(0)

    def assertListAlmostEqual(self, arr, brr, places=6):
        for a, b in zip(list(arr), list(brr)):
            self.assertAlmostEqual(float(a), float(b), places)
