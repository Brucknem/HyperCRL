import copy
import math
import shutil
import unittest
import numpy as np
import torch

from hypercrl.srl.datautil import DataCollector, DataPoint
from hypercrl.srl.utils import convert_to_array
from hypercrl.tools.default_arg import Hparams
from tests.test_base import TestBase
from signal import *
import sys, time

save_path = "tmp/"


def clean_dirs():
    shutil.rmtree(save_path, ignore_errors=True)


class TestDataUtil(TestBase):
    """
    Unittests for the robotic priors
    """

    def generate_datapoint(self):
        val = self.datapoint_index
        x_t = np.array([val] * 512)
        u = np.array([val] * 7)
        r = val
        gt_x_t = np.array([val] * 26)
        self.datapoint_index += 1

        return DataPoint(features=x_t, action=u, next_features=x_t, reward=r, real_state=gt_x_t, next_real_state=gt_x_t)

    def setUp(self) -> None:
        np.random.seed(int('0b10101010101010101010101010101', 2))

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

        self.datapoint_index = 0

    def test_add(self) -> None:
        collector = DataCollector(hparams=self.hparams)
        task_id = 0

        for i in range(self.hparams.vision_params.collector_max_capacity):
            datapoint = self.generate_datapoint()
            collector.add(task_id, *datapoint)
            self.assertEqual(len(collector.data_points[task_id]), i + 1)

        self.assertListEqual(collector.train_inds[task_id],
                             [True, True, False, True, True, False, False, True, True, True])

    def test_delete_on_max_capacity(self) -> None:
        collector = DataCollector(hparams=self.hparams)
        task_id = 0

        for i in range(self.hparams.vision_params.collector_max_capacity):
            datapoint = self.generate_datapoint()
            collector.add(task_id, *datapoint)

        for i in range(self.hparams.vision_params.collector_max_capacity):
            datapoint = self.generate_datapoint()
            collector.add(task_id, *datapoint)
            self.assertEqual(len(collector.data_points[task_id]), self.hparams.vision_params.collector_max_capacity)

        self.assertListEqual(collector.train_inds[task_id],
                             [True, False, True, True, True, True, False, True, True, True])

        datapoint = self.generate_datapoint()
        for i in range(500):
            collector.add(task_id, *datapoint)

        self.assertListEqual(collector.train_inds[task_id],
                             [True, True, False, True, False, True, True, True, True, True])

    def test_sample_action(self) -> None:
        collector = DataCollector(hparams=self.hparams)
        task_id = 0

        for i in range(self.hparams.vision_params.collector_max_capacity):
            datapoint = self.generate_datapoint()
            collector.add(task_id, *datapoint)

        for _ in range(100):
            self.assertTrue(collector.check_same_actions_really_same(task_id))

            action = collector.sample_action(task_id)
            expected_action = collector.data_points[task_id][collector.same_actions[task_id][tuple(action)][0]].action
            self.assertListAlmostEqual(expected_action, action)
            collector.add(task_id, *self.generate_datapoint())

    def test_merge(self):
        task_id = 0
        other_task_id = task_id + 1

        self.hparams.vision_params.collector_max_capacity = -1

        a = DataCollector(hparams=self.hparams)
        b = DataCollector(hparams=self.hparams)
        for i in range(10):
            data_point = self.generate_datapoint()
            a.add(task_id, *data_point)
            b.add(task_id, *data_point)
            b.add(task_id, *self.generate_datapoint())
            b.add(other_task_id, *self.generate_datapoint())

        a.merge(b)

        self.assertEqual(a.data_points[other_task_id], b.data_points[other_task_id])
        self.assertEqual(a.train_inds[other_task_id], b.train_inds[other_task_id])
        self.assertEqual(a.same_actions[other_task_id], b.same_actions[other_task_id])

        self.assertEqual(a.data_points[task_id][-len(b.data_points[task_id]):], b.data_points[task_id])
        self.assertEqual(a.train_inds[task_id][-len(b.train_inds[task_id]):], b.train_inds[task_id])

    def test_save_and_load(self) -> None:
        self.hparams.vision_params.collector_max_capacity = -1
        self.hparams.vision_params.load_max = -1
        self.hparams.vision_params.save_every = 100

        collector = DataCollector(hparams=self.hparams)
        task_id = 0
        clean_dirs()

        for i in range(100):
            collector.add(task_id, *(self.generate_datapoint()))

        backup_collector = copy.deepcopy(collector)
        collector.save()

        collector = DataCollector(hparams=self.hparams)
        collector.load()

        self.assertEqual(backup_collector, collector)
        clean_dirs()


if __name__ == '__main__':
    unittest.main()
