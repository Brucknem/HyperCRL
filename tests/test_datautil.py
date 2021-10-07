import math
import unittest
import numpy as np
import torch

from hypercrl.srl.datautil import DataCollector
from hypercrl.tools.default_arg import Hparams, VisionParams


def generate_datapoint(width, height):
    x_t = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    u = np.random.rand(7) * 2 - 1
    r = np.random.rand() * 20 - 10
    return x_t, u, x_t, r


class TestDataUtil(unittest.TestCase):
    """
    Unittests for the robotic priors
    """

    def setUp(self) -> None:
        np.random.seed(0)

        self.save_path = "tmp/"

        self.hparams = Hparams()
        self.hparams.dnn_out = True
        self.hparams.normalize_xu = True
        self.hparams.env = "door_pose"

        self.hparams.vision_params = VisionParams()
        self.width = 32
        self.height = 32
        self.hparams.vision_params.camera_widths = self.width
        self.hparams.vision_params.camera_heights = self.height
        self.hparams.vision_params.collector_max_capacity = 10
        self.hparams.vision_params.save_every = -1
        self.hparams.vision_params.load_max = 100
        self.hparams.vision_params.load_suffix = "x"
        self.hparams.vision_params.save_path = self.save_path

    def test_add(self) -> None:
        collector = DataCollector(hparams=self.hparams)
        task_id = 0

        for i in range(self.hparams.vision_params.collector_max_capacity):
            datapoint = generate_datapoint(self.width, self.height)
            collector.add(*datapoint, 0)
            self.assertEqual(len(collector.images[task_id]), i + 1)
            self.assertEqual(len(collector.nexts[task_id]), i + 1)
            self.assertEqual(len(collector.actions[task_id]), i + 1)
            self.assertEqual(len(collector.rewards[task_id]), i + 1)

        self.assertListEqual(collector.train_inds[task_id], [1, 2, 3, 4, 6, 7, 8])
        self.assertListEqual(collector.val_inds[task_id], [0, 5, 9])

    def test_delete_on_max_capacity(self) -> None:
        collector = DataCollector(hparams=self.hparams)
        task_id = 0

        for i in range(self.hparams.vision_params.collector_max_capacity):
            datapoint = generate_datapoint(self.width, self.height)
            collector.add(*datapoint, 0)

        for i in range(self.hparams.vision_params.collector_max_capacity):
            datapoint = generate_datapoint(self.width, self.height)
            collector.add(*datapoint, 0)
            self.assertEqual(len(collector.images[task_id]), self.hparams.vision_params.collector_max_capacity)
            self.assertEqual(len(collector.nexts[task_id]), self.hparams.vision_params.collector_max_capacity)
            self.assertEqual(len(collector.actions[task_id]), self.hparams.vision_params.collector_max_capacity)
            self.assertEqual(len(collector.rewards[task_id]), self.hparams.vision_params.collector_max_capacity)

        self.assertListEqual(collector.train_inds[task_id], [1, 2, 3, 4, 6, 7, 8])
        self.assertListEqual(collector.val_inds[task_id], [0, 5, 9])


if __name__ == '__main__':
    unittest.main()
