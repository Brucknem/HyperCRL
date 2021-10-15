import copy
import math
import shutil
import unittest
import numpy as np
import torch

from hypercrl.srl.datautil import DataCollector
from hypercrl.srl.utils import convert_to_array
from hypercrl.tools.default_arg import Hparams
from tests.test_base import TestBase
from signal import *
import sys, time

save_path = "tmp/"


def clean_dirs():
    shutil.rmtree(save_path, ignore_errors=True)


def generate_datapoint(width, height):
    x_t = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
    u = np.random.rand(7) * 2 - 1
    r = np.random.rand() * 20 - 10
    gt_x_t = np.random.rand(26)
    gt_x_tt = np.random.rand(26)
    return x_t, u, x_t, r, gt_x_t, gt_x_tt


class TestDataUtil(TestBase):
    """
    Unittests for the robotic priors
    """

    def setUp(self) -> None:
        np.random.seed(0)

        self.save_path = save_path

        self.hparams = Hparams()
        self.hparams.dnn_out = True
        self.hparams.normalize_xu = True
        self.hparams.env = "door_pose"

        self.hparams.vision_params = Hparams()
        self.width = 32
        self.height = 32
        self.hparams.vision_params.camera_widths = self.width
        self.hparams.vision_params.camera_heights = self.height
        self.hparams.vision_params.collector_max_capacity = 10
        self.hparams.vision_params.save_every = -1
        self.hparams.vision_params.load_max = 100
        self.hparams.vision_params.load_suffix = "x"
        self.hparams.vision_params.save_suffix = "x"

        self.hparams.vision_params.save_path = self.save_path

    def test_add(self) -> None:
        collector = DataCollector(hparams=self.hparams)
        task_id = 0

        for i in range(self.hparams.vision_params.collector_max_capacity):
            datapoint = generate_datapoint(self.width, self.height)
            collector.add(task_id, *datapoint)
            self.assertEqual(len(collector.images[task_id]), i + 1)
            self.assertEqual(len(collector.nexts[task_id]), i + 1)
            self.assertEqual(len(collector.actions[task_id]), i + 1)
            self.assertEqual(len(collector.rewards[task_id]), i + 1)
            self.assertEqual(len(collector.gt_state[task_id]), i + 1)
            self.assertEqual(len(collector.gt_nexts[task_id]), i + 1)

        self.assertListEqual(collector.train_inds[task_id], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertListEqual(collector.val_inds[task_id], [])

    def test_delete_on_max_capacity(self) -> None:
        collector = DataCollector(hparams=self.hparams)
        task_id = 0

        for i in range(self.hparams.vision_params.collector_max_capacity):
            datapoint = generate_datapoint(self.width, self.height)
            collector.add(task_id, *datapoint)

        datapoint = generate_datapoint(self.width, self.height)
        collector.add(task_id, *datapoint)
        self.assertListEqual(collector.train_inds[task_id], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertListEqual(collector.val_inds[task_id], [])

        for i in range(self.hparams.vision_params.collector_max_capacity):
            datapoint = generate_datapoint(self.width, self.height)
            collector.add(task_id, *datapoint)
            self.assertEqual(len(collector.images[task_id]), self.hparams.vision_params.collector_max_capacity)
            self.assertEqual(len(collector.nexts[task_id]), self.hparams.vision_params.collector_max_capacity)
            self.assertEqual(len(collector.actions[task_id]), self.hparams.vision_params.collector_max_capacity)
            self.assertEqual(len(collector.rewards[task_id]), self.hparams.vision_params.collector_max_capacity)

        self.assertListEqual(collector.train_inds[task_id], [0, 1, 2, 3, 4, 5, 9])
        self.assertListEqual(collector.val_inds[task_id], [6, 7, 8])

        datapoint = generate_datapoint(self.width, self.height)
        for i in range(50):
            collector.add(task_id, *datapoint)

        self.assertListEqual(collector.train_inds[task_id], [0, 1, 2, 3, 6, 7, 8])
        self.assertListEqual(collector.val_inds[task_id], [4, 5, 9])

    def test_sample_action(self) -> None:
        collector = DataCollector(hparams=self.hparams)
        task_id = 0

        for i in range(self.hparams.vision_params.collector_max_capacity):
            datapoint = generate_datapoint(self.width, self.height)
            collector.add(task_id, *datapoint)

        for _ in range(100):
            self.assertTrue(collector.check_same_actions_really_same(task_id))

            action = collector.sample_action(task_id)
            expected_action = collector.actions[task_id][collector.same_actions[task_id][tuple(action)][0]]
            self.assertListAlmostEqual(expected_action, action)
            collector.add(task_id, *generate_datapoint(self.width, self.height))

    def test_save_and_load(self) -> None:
        self.hparams.vision_params.collector_max_capacity = -1
        self.hparams.vision_params.load_max = -1
        self.hparams.vision_params.save_every = 100

        collector = DataCollector(hparams=self.hparams)
        task_id = 0
        clean_dirs()

        datapoints = []
        for i in range(10):
            for _ in range(2):
                datapoint = generate_datapoint(self.width, self.height)
                datapoints.append(datapoint)
                collector.add(task_id, *datapoint)

            collector.save()

        collector = DataCollector(hparams=self.hparams)
        collector.load()
        
        for i, datapoint in enumerate(datapoints):
            x_t, u, x_tt, r, gt_x_t, gt_x_tt = datapoint
            self.assertListAlmostEqual(collector.actions[task_id][i], u)
            self.assertAlmostEqual(collector.rewards[task_id][i], r)
            self.assertListAlmostEqual(collector.gt_state[task_id][i], gt_x_t)
            self.assertListAlmostEqual(collector.gt_nexts[task_id][i], gt_x_tt)

            self.assertTrue(np.array_equal(collector.images[task_id][i], x_t.reshape((3, self.width, self.height))))
            self.assertTrue(np.array_equal(collector.nexts[task_id][i], x_tt.reshape((3, self.width, self.height))))

        clean_dirs()
        collector = DataCollector(hparams=self.hparams)
        for i in range(self.hparams.vision_params.save_every):
            x_t, u, x_tt, r, gt_x_t, gt_x_tt = generate_datapoint(self.width, self.height)
            if not collector.empty(task_id) and np.random.rand() < 0.8:
                u = collector.sample_action(task_id)

            collector.add(task_id, x_t, u, x_tt, r, gt_x_t, gt_x_tt)

        backup_collector = copy.deepcopy(collector)
        clean_dirs()
        collector.save()

        collector = DataCollector(hparams=self.hparams)
        collector.load()

        self.assertEqual(backup_collector, collector)

        collector.save()
        collector = DataCollector(hparams=self.hparams)
        collector.load()

        for i in range(self.hparams.vision_params.save_every):
            self.assertTrue(np.array_equal(collector.images[task_id][i], collector.images[task_id][i + 100]))
            self.assertTrue(np.array_equal(collector.nexts[task_id][i], collector.nexts[task_id][i + 100]))
            self.assertTrue(np.array_equal(collector.rewards[task_id][i], collector.rewards[task_id][i + 100]))
            self.assertTrue(np.array_equal(collector.actions[task_id][i], collector.actions[task_id][i + 100]))
            self.assertTrue(np.array_equal(collector.gt_state[task_id][i], collector.gt_state[task_id][i + 100]))
            self.assertTrue(np.array_equal(collector.gt_nexts[task_id][i], collector.gt_nexts[task_id][i + 100]))

        clean_dirs()


if __name__ == '__main__':
    unittest.main()
