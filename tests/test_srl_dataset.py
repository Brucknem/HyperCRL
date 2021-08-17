import math
import unittest

import numpy as np
import torch
from torchvision import transforms

from hypercrl.srl.srl_dataset import SRLDataSet


class TestSRLDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset: SRLDataSet = SRLDataSet(transform=None)

        self.dimension = (128, 128, 3)
        self.dataset.add_datapoint(0, np.zeros(self.dimension), np.zeros(7), 1)
        self.dataset.add_datapoint(0, np.ones(self.dimension), np.ones(7), 0)
        self.dataset.add_datapoint(0, np.ones(self.dimension), np.zeros(7), 1)
        self.dataset.add_datapoint(0, np.ones(self.dimension), np.ones(7), 0)
        self.dataset.add_datapoint(0, np.zeros(self.dimension), np.zeros(7), 1)
        self.dataset.calculate_same_action_pairs()

        to_tensor = transforms.ToTensor()
        self.zero_tensor = to_tensor(np.zeros(self.dimension))
        self.one_tensor = to_tensor(np.ones(self.dimension))

    def test_add(self):
        self.assertEqual(self.dataset.data_points[0].episode, 0)
        self.assertTrue(np.array_equal(self.dataset.data_points[0].observation, np.zeros(self.dimension)))
        self.assertListEqual(list(self.dataset.data_points[0].action), list(np.zeros(7)))
        self.assertEqual(self.dataset.data_points[0].reward, 1)

    def test_same_action_indices(self):
        same_actions = self.dataset.same_action_pairs

        self.assertEqual(len(same_actions), 4)
        self.assertIn(((0, 1), (2, 3)), same_actions)
        self.assertIn(((2, 3), (0, 1)), same_actions)
        self.assertIn(((1, 2), (3, 4)), same_actions)
        self.assertIn(((3, 4), (1, 2)), same_actions)

    def test_get(self):
        self.assertEqual(len(self.dataset), 4)

        entry = self.dataset[0]

        self.assertTrue(torch.equal(entry['observations'][0][0], self.zero_tensor))
        self.assertTrue(torch.equal(entry['observations'][0][1], self.one_tensor))
        self.assertTrue(torch.equal(entry['observations'][1][0], self.one_tensor))
        self.assertTrue(torch.equal(entry['observations'][1][1], self.one_tensor))

        self.assertTrue(torch.equal(entry['actions'][0][0], torch.from_numpy(np.zeros(7))))
        self.assertTrue(torch.equal(entry['actions'][0][1], torch.from_numpy(np.ones(7))))
        self.assertTrue(torch.equal(entry['actions'][1][0], torch.from_numpy(np.zeros(7))))
        self.assertTrue(torch.equal(entry['actions'][1][1], torch.from_numpy(np.ones(7))))

        self.assertTrue(torch.equal(entry['rewards'][0][0], torch.tensor(1)))
        self.assertTrue(torch.equal(entry['rewards'][0][1], torch.tensor(0)))
        self.assertTrue(torch.equal(entry['rewards'][1][0], torch.tensor(1)))
        self.assertTrue(torch.equal(entry['rewards'][1][1], torch.tensor(0)))
