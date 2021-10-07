import unittest
import numpy as np

from hypercrl.srl.utils import sample_by_size, probabilities_by_size


class TestDataUtil(unittest.TestCase):
    """
    Unittests for the utils
    """

    def setUp(self) -> None:
        np.random.seed(0)

    def test_probabilities_by_size(self):
        arr = np.array(range(10))

        expected = [0.2, 0.17777778, 0.15555556, 0.13333333, 0.11111111,
                    0.08888889, 0.06666667, 0.04444444, 0.02222222, 0.]

        probabilities = probabilities_by_size(arr, inverse=True, lower_as_median=False)
        self.assertListAlmostEqual(list(probabilities), expected, 6)
        probabilities = probabilities_by_size(arr, inverse=False, lower_as_median=False)
        self.assertListAlmostEqual(list(probabilities), expected[::-1], 6)

        expected = [0.25714286, 0.22857143, 0.2, 0.17142857, 0.14285714, 0, 0, 0, 0, 0]
        probabilities = probabilities_by_size(arr, inverse=True, lower_as_median=True)
        self.assertListAlmostEqual(list(probabilities), expected, 6)
        probabilities = probabilities_by_size(arr, inverse=False, lower_as_median=True)
        self.assertListAlmostEqual(list(probabilities), expected[::-1], 6)

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

    def assertListAlmostEqual(self, arr, brr, places=6):
        for a, b in zip(arr, brr):
            self.assertAlmostEqual(a, b, places)


if __name__ == '__main__':
    unittest.main()
