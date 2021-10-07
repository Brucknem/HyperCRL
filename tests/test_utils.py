import unittest
import numpy as np

from hypercrl.srl.utils import sample_by_size, probabilities_by_size, remove_and_move
from tests.test_base import TestBase


class TestDataUtil(TestBase):
    """
    Unittests for the utils
    """

    def setUp(self) -> None:
        np.random.seed(0)

    def test_probabilities_by_size(self):
        self.assertListEqual([1], list(probabilities_by_size([1])))

        arr = np.array(range(10))

        expected = [0.2, 0.17777778, 0.15555556, 0.13333333, 0.11111111,
                    0.08888889, 0.06666667, 0.04444444, 0.02222222, 0.]

        probabilities = probabilities_by_size(arr, inverse=True, lower_as_median=False)
        self.assertListAlmostEqual(probabilities, expected, 6)
        probabilities = probabilities_by_size(arr, inverse=False, lower_as_median=False)
        self.assertListAlmostEqual(probabilities, expected[::-1], 6)

        expected = [0.25714286, 0.22857143, 0.2, 0.17142857, 0.14285714, 0, 0, 0, 0, 0]
        probabilities = probabilities_by_size(arr, inverse=True, lower_as_median=True)
        self.assertListAlmostEqual(probabilities, expected, 6)
        probabilities = probabilities_by_size(arr, inverse=False, lower_as_median=True)
        self.assertListAlmostEqual(probabilities, expected[::-1], 6)

        probabilities = probabilities_by_size([1] * 10)
        self.assertListEqual(list(probabilities), [0.1] * 10)

    def test_sample_by_size(self):
        arr = np.array(range(10))

        idx = sample_by_size(arr, inverse=True, lower_as_median=False)
        self.assertEqual(idx, 3)

        idx = sample_by_size(arr, inverse=True, lower_as_median=True)
        self.assertEqual(idx, 3)
        for _ in range(50):
            self.assertLess(sample_by_size(arr, inverse=True, lower_as_median=True), 5)

        idx = sample_by_size(arr, inverse=False, lower_as_median=False)
        self.assertEqual(idx, 9)

        idx = sample_by_size(arr, inverse=False, lower_as_median=True)
        self.assertEqual(idx, 5)
        for _ in range(50):
            self.assertGreaterEqual(sample_by_size(arr, inverse=False, lower_as_median=True), 5)

        self.assertEqual(0, sample_by_size([1], inverse=True, lower_as_median=False))

    def test_remove_and_move(self):
        arr = np.array(range(0, 20, 2))
        arr = remove_and_move(arr, 10)
        self.assertListEqual(arr, list(range(0, 10, 2)) + list(range(11, 19, 2)))

        arr = remove_and_move(arr, 10)
        self.assertListEqual(arr, list(range(0, 10, 2)) + list(range(10, 18, 2)))


if __name__ == '__main__':
    unittest.main()
