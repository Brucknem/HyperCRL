from collections import defaultdict
from functools import lru_cache

import cv2
import numpy as np
import time
import torch
from torch.utils.data import TensorDataset

import hypercrl.dataset.datautil
from hypercrl.hypercl import HyperNetwork
from hypercrl.srl import ResNet18EncoderHnet
from hypercrl.srl.utils import timeit_context

import pathlib
import pandas as pd

from hypercrl.srl.utils import sample_by_size, remove_and_move, probabilities_by_size, convert_to_array


def shorten_array(arr):
    if len(arr) <= 3:
        return str(arr)
    else:
        return f'[{arr[0]:.4f}, {arr[1]:.4f}, ...]'


class DataPoint:
    features_in_datapoint = ['features', 'action', 'next_features', 'reward', 'real_state', 'next_real_state']

    def __init__(self, features, action, next_features, reward, real_state=None, next_real_state=None):
        self.features = features
        self.action = action
        self.next_features = next_features
        self.reward = reward
        self.real_state = real_state
        self.next_real_state = next_real_state

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if other == 0:
            return self

        if not isinstance(other, DataPoint):
            raise f"{other} is not a datapoint."

        return DataPoint(
            self.features + other.features,
            self.action + other.action,
            self.next_features + other.next_features,
            self.reward + other.reward,
            self.real_state + other.real_state,
            self.next_real_state + other.next_real_state,
        )

    def __eq__(self, other):
        return isinstance(other, DataPoint) and \
               np.array_equal(self.features, other.features, equal_nan=True) and \
               np.array_equal(self.action, other.action, equal_nan=True) and \
               np.array_equal(self.next_features, other.next_features, equal_nan=True) and \
               np.array_equal(self.reward, other.reward, equal_nan=True) and \
               np.array_equal(self.real_state, other.real_state, equal_nan=True) and \
               np.array_equal(self.next_real_state, other.next_real_state, equal_nan=True)

    def __repr__(self):
        return f'features: {shorten_array(self.features)}, ' \
               f'action: {shorten_array(self.action)}, ' \
               f'next_features: {shorten_array(self.next_features)}, ' \
               f'reward: {self.reward:.4f}, ' \
               f'real_state: {shorten_array(self.real_state)}, ' \
               f'next_real_state: {shorten_array(self.next_real_state)}'

    def __getitem__(self, item):
        if item == 0 or item == 'features':
            return self.features
        if item == 1 or item == 'action':
            return self.action
        if item == 2 or item == 'next_features':
            return self.next_features
        if item == 3 or item == 'reward':
            return self.reward
        if item == 4 or item == 'real_state':
            return self.real_state
        if item == 5 or item == 'next_real_state':
            return self.next_real_state
        raise IndexError(f'{item} not in datapoint')


class DataCollector:
    """
    Image data collector for the SRL module
    """

    features_to_write = ['features', 'action', 'next_features', 'reward', 'real_state', 'next_real_state', 'train_inds']

    def __init__(self, hparams):
        self.data_points = {}

        self.train_inds = {}
        self.train_inds_updated = {}
        self.train_inds_cache = {}
        self.train_inds_cache_set = {}
        self.val_inds_cache = {}
        self.val_inds_cache_set = {}

        self.same_actions = {}

        self.max_capacity = hparams.vision_params.collector_max_capacity

        self.save_path = hparams.vision_params.save_path
        self.save_every = hparams.vision_params.save_every

        self.load_max = hparams.vision_params.load_max
        self.save_suffix = hparams.vision_params.save_suffix
        self.load_suffix = hparams.vision_params.load_suffix

        self.hparams = hparams

        self.mins = {}
        self.maxs = {}
        self.ranges = {}

    def calculate_statistics(self, task_id):
        values = {x: self.get_values(task_id, x) for x in DataPoint.features_in_datapoint}
        self.mins[task_id] = {x: np.amin(value, axis=0) for x, value in values.items()}
        self.maxs[task_id] = {x: np.amax(value, axis=0) for x, value in values.items()}
        self.ranges[task_id] = {x: self.maxs[task_id][x] - self.mins[task_id][x] for x in values.keys()}

    def num_tasks(self):
        return len(self.data_points)

    def add(self, task_id, features, action, next_features, reward, real_state=None, next_real_state=None):
        # Convert Format
        features, action, next_features, real_state, next_real_state = \
            (convert_to_array(x) for x in [features, action, next_features, real_state, next_real_state])

        datapoint = DataPoint(features=features, action=action, next_features=next_features, reward=reward,
                              real_state=real_state, next_real_state=next_real_state)

        if task_id not in self.data_points:
            self.data_points[task_id] = []
        self.data_points[task_id].append(datapoint)

        if task_id not in self.train_inds:
            self.train_inds[task_id] = []
        self.train_inds[task_id].append(np.random.rand() < 0.75)
        self.train_inds_updated[task_id] = True

        ind = len(self.data_points[task_id]) - 1

        if task_id not in self.same_actions:
            self.same_actions[task_id] = {}
        action = tuple(np.squeeze(action))
        if action not in self.same_actions[task_id]:
            self.same_actions[task_id][action] = []
        self.same_actions[task_id][action].append(ind)

        self.delete_on_max_capacity(task_id)

    def update_inds_cache(self, task_id):
        if self.train_inds_updated[task_id]:
            self.train_inds_cache[task_id] = np.where(self.train_inds[task_id])[0]
            self.val_inds_cache[task_id] = np.where(~np.array(self.train_inds[task_id]))[0]
            self.train_inds_cache_set[task_id] = set(self.train_inds_cache[task_id])
            self.val_inds_cache_set[task_id] = set(self.val_inds_cache[task_id])
            self.train_inds_updated[task_id] = False

    def get_train_inds(self, task_id, as_set=False):
        self.update_inds_cache(task_id)
        if as_set:
            return self.train_inds_cache_set[task_id]
        return self.train_inds_cache[task_id]

    def get_val_inds(self, task_id, as_set=False):
        self.update_inds_cache(task_id)
        if as_set:
            return self.val_inds_cache_set[task_id]
        return self.val_inds_cache[task_id]

    def get_features(self, task_id, idx=-1):
        if idx >= 0:
            return self.data_points[task_id][idx].features
        return np.array([x.features for x in self.data_points[task_id]])

    def get_actions(self, task_id, idx=-1):
        if idx >= 0:
            return self.data_points[task_id][idx].action
        return np.array([x.action for x in self.data_points[task_id]])

    def get_next_features(self, task_id, idx=-1):
        if idx >= 0:
            return self.data_points[task_id][idx].next_features
        return np.array([x.next_features for x in self.data_points[task_id]])

    def get_rewards(self, task_id, idx=-1):
        if idx >= 0:
            return self.data_points[task_id][idx].reward
        return np.array([x.reward for x in self.data_points[task_id]])

    def get_real_states(self, task_id, idx=-1):
        if idx >= 0:
            return self.data_points[task_id][idx].real_state
        return np.array([x.real_state for x in self.data_points[task_id]])

    def get_next_real_state(self, task_id, idx=-1):
        if idx >= 0:
            return self.data_points[task_id][idx].next_real_state
        return np.array([x.next_real_state for x in self.data_points[task_id]])

    def get_values(self, task_id, item):
        if item == 'train_inds':
            return self.train_inds[task_id]
        return np.array([x[item] for x in self.data_points[task_id]])

    def delete_on_max_capacity(self, task_id):
        if self.max_capacity <= 0:
            return

        while len(self.data_points[task_id]) > self.max_capacity:
            remove_from_train = sum(self.train_inds[task_id]) >= 0.75 * len(self.train_inds[task_id])
            remove_from = self.get_train_inds(task_id) if remove_from_train else self.get_val_inds(task_id)

            idx = sample_by_size(self.get_rewards(task_id)[remove_from], lower_as_median=True)
            idx = remove_from[idx]

            del self.train_inds[task_id][idx]
            del self.data_points[task_id][idx]
            self.train_inds_updated[task_id] = True

            to_del = []
            for key in self.same_actions[task_id].keys():
                self.same_actions[task_id][key] = remove_and_move(self.same_actions[task_id][key], idx)
                if len(self.same_actions[task_id][key]) == 0:
                    to_del.append(key)

            for key in to_del:
                del self.same_actions[task_id][key]

    def get_dataset(self, task_id, ds_range=None):
        """
        Return a pytorch dataset of (state, actions, next_state)
        states, actions are normalized to N(0, 1)
        """

        values = [self.get_values(task_id, x) for x in DataPoint.features_in_datapoint]
        values = [torch.FloatTensor(value) for value in values]
        values = [torch.IntTensor(list(range(len(self.data_points[task_id]))))] + values

        train_inds = self.get_train_inds(task_id)
        val_inds = self.get_val_inds(task_id)

        train_set = TensorDataset(*([value[train_inds] for value in values]))
        val_set = TensorDataset(*([value[val_inds] for value in values]))

        return train_set, val_set

    def get_whole_dataset(self, task_id):
        indices = torch.IntTensor(list(range(len(self.images[task_id]))))
        images = torch.FloatTensor(np.hstack(self.images[task_id])).reshape(self.image_dims)
        actions = torch.FloatTensor(np.hstack(self.actions[task_id])).T
        nexts = torch.FloatTensor(np.hstack(self.nexts[task_id])).reshape(self.image_dims)

        train_set = TensorDataset(indices, images, actions, nexts)

        return train_set

    def get_same_actions(self, task_id, idx):
        train = self.train_inds[task_id][idx]
        indices = self.get_train_inds(task_id, True) if train else self.get_val_inds(task_id, True)
        action = tuple(self.data_points[task_id][idx].action)
        same_action_indices = list(set(self.same_actions[task_id][action]) & indices)

        same_action_indices.remove(idx)

        data_points = [self.data_points[task_id][x] for x in same_action_indices]
        return data_points

    def get_by_action(self, task_id, u):
        if isinstance(u, torch.Tensor):
            u = u.detach().cpu().numpy()
        if u.ndim == 1:
            u = u[:, None]
        u = np.squeeze(u)

        ts = time.time()
        keys = list(self.same_actions[task_id].keys())
        indices = np.where(np.linalg.norm(np.array(keys) - u, axis=-1) < 1e-5)
        indices = [self.same_actions[task_id][keys[i[0]]] for i in indices]
        # print(f'Search same actions time: {time.time() - ts}')
        if not indices:
            return torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

        ts = time.time()
        indices = np.squeeze(indices)

        images = torch.from_numpy(np.array(self.images[task_id]))[indices].reshape(self.image_dims)
        actions = torch.from_numpy(np.array(self.actions[task_id]))[indices]
        nexts = torch.from_numpy(np.array(self.nexts[task_id]))[indices].reshape(self.image_dims)
        # print(f'Create Tensors time: {time.time() - ts}')

        return images, actions, nexts

    def convert(self, task_id: int, encoder: ResNet18EncoderHnet, hnet: HyperNetwork,
                collector: hypercrl.dataset.datautil.DataCollector, gpuid: str):
        collector.clear()

        train_loader = torch.utils.data.DataLoader(self.get_whole_dataset(task_id), batch_size=64,
                                                   shuffle=True, drop_last=False, num_workers=0)
        encoder.to(gpuid)
        hnet.to(gpuid)

        encoder.eval()
        hnet.eval()

        encoder_weights = hnet.forward(task_id)

        for i, data in enumerate(train_loader):
            states = encoder.forward(data[1].to(gpuid), encoder_weights).detach().cpu().numpy()
            nexts = encoder.forward(data[3].to(gpuid), encoder_weights).detach().cpu().numpy()

            for x_t, u, x_tt in zip(states, data[2], nexts):
                collector.add(x_t, u, x_tt, task_id)

    def sample_action(self, task_id):
        least_same_action_indices = {key: len(value) for (key, value) in self.same_actions[task_id].items()}
        probs = np.array(list(least_same_action_indices.values()))

        idx = sample_by_size(probs, inverse=True, lower_as_median=True)
        action = list(least_same_action_indices.keys())[idx]
        # print(f'Sampled action: {idx} -> {action}')
        return np.array(action)

    def empty(self, task_id):
        return task_id not in self.data_points or len(self.data_points[task_id]) == 0

    def merge(self, other):
        for task_id in self.data_points:
            if not other.empty(task_id):
                old_len = len(self.data_points[task_id])
                self.data_points[task_id] += other.data_points[task_id]
                self.train_inds[task_id] += other.train_inds[task_id]
                for key, value in other.same_actions[task_id].items():
                    if key not in self.same_actions[task_id]:
                        self.same_actions[task_id][key] = []
                    self.same_actions[task_id][key] += list(np.array(value) + old_len)
                self.check_same_actions_really_same(task_id)
                self.train_inds_updated[task_id] = True

        for task_id in other.data_points:
            if self.empty(task_id):
                self.data_points[task_id] = other.data_points[task_id]
                self.train_inds[task_id] = other.train_inds[task_id]
                self.same_actions[task_id] = other.same_actions[task_id]
                self.train_inds_updated[task_id] = True

    def save(self):
        for task_id in self.data_points:
            if self.empty(task_id):
                continue
            save_path = pathlib.Path(self.save_path).joinpath(str(self.save_suffix)).joinpath(str(task_id))
            save_path.mkdir(parents=True, exist_ok=True)

            # MASTER_THESIS Load old when saving
            # load_collector = DataCollector(self.hparams)
            # load_collector.load_task(task_id)
            #
            # self.merge(load_collector)

            new_values = [pd.DataFrame(self.get_values(task_id, feature)) for feature in
                          DataCollector.features_to_write]
            paths = [save_path.joinpath(f'{path}.csv') for path in DataCollector.features_to_write]
            for path, values in zip(paths, new_values):
                values.to_csv(path, index=False, header=False)

    def clear(self, task_id):
        self.data_points[task_id] = []
        self.train_inds[task_id] = []
        self.train_inds_updated[task_id] = True

    def load(self):
        load_path_base = pathlib.Path(self.save_path).joinpath(self.load_suffix)
        for task_id in [x.name for x in load_path_base.iterdir() if x.is_dir()]:
            self.load_task(int(task_id))
            self.calculate_statistics(int(task_id))

    def load_task(self, task_id):
        self.clear(task_id)
        load_path = pathlib.Path(self.save_path).joinpath(self.load_suffix).joinpath(str(task_id))

        if not load_path.exists():
            return

        load_paths = [pathlib.Path(f'{load_path.joinpath(path)}.csv') for path in DataCollector.features_to_write]
        load_paths = [path if path.exists() else None for path in load_paths]
        if None in load_paths:
            return

        values = [pd.read_csv(path, index_col=False, header=None).to_numpy() for path in load_paths]
        values[6] = [True if x else False for x in values[6]]

        for i in range(len(values[0])):
            datapoint = DataPoint(features=values[0][i], action=values[1][i], next_features=values[2][i],
                                  reward=float(values[3][i]), real_state=values[4][i], next_real_state=values[5][i])
            self.add(task_id, *datapoint)
            self.train_inds[task_id][-1] = values[6][i]

        # assert self.check_same_actions_really_same(task_id)
        self.train_inds_updated[task_id] = True
        self.normalize(task_id)

    def normalize(self, task_id):
        features = self.get_values(task_id, 'features')
        next_features = self.get_values(task_id, 'next_features')

        features_mean = np.mean(features, axis=0)
        next_features_mean = np.mean(next_features, axis=0)
        features_std = np.std(features, axis=0)
        next_features_std = np.std(next_features, axis=0)

        for i in range(len(self.data_points[task_id])):
            self.data_points[task_id][i].features = (self.data_points[task_id][
                                                         i].features - features_mean) / features_std
            self.data_points[task_id][i].next_features = (self.data_points[task_id][
                                                              i].next_features - next_features_mean) / next_features_std

    def __eq__(self, other):
        if not isinstance(other, DataCollector):
            return False

        return self.data_points == other.data_points and \
               self.train_inds == other.train_inds and \
               self.same_actions == other.same_actions

    def check_same_actions_really_same(self, task_id):
        for key, value in self.same_actions[task_id].items():
            for idx in value:
                for a, b in zip(list(key), list(self.get_actions(task_id)[idx])):
                    if not np.allclose(a, b):
                        return False
        return True
