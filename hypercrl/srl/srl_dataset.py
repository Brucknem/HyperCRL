import copy
import math
import os
import pathlib
import sys
import time
from typing import Union

import numpy as np
import torch
import torchvision
import yaml
from torchvision import transforms
from torch.utils.data import Dataset

from hypercrl.srl import SRL
from collections import defaultdict
import cv2
from yaml import load, dump, full_load
import json


class DataPoint:
    """
    Wrapper for a datapoint within an episode.
    """

    def __init__(self, episode: int, step: int, observation: np.ndarray, action: np.ndarray, reward: float):
        """
        Args:
            episode: The id of the episode.
            observation: The raw image observation.
            action: The action taken after the observations was recorded.
            reward: The reward at the observation.
        """
        self.step = step
        self.episode = episode
        self.observation = observation
        self.action = action
        self.reward = float(reward)


class SRLDataSet(Dataset):
    """
    The state representation learning dataset.
    """

    def __init__(self, horizon: int, seed: int = 12345,
                 persistence_dir: str = None, add_timestamp_folder: bool = True):
        """
        Args:
            transform: A transformation that is applied to the raw image observation.
                        If set to 'default' subtract mean and divide by variance of ImageNet dataset.
            seed: The random seed for numpy.
            persistence_dir: A directory to save the observations.
        """
        self.data_points = []

        self.seed = seed
        np.random.seed(self.seed)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.same_action_buffer = defaultdict(set)
        self.same_action_pairs = []

        self.horizon = horizon

        self.persistence_dir = None
        if persistence_dir:
            self.persistence_dir = pathlib.Path(persistence_dir)
            if add_timestamp_folder:
                self.persistence_dir = self.persistence_dir.joinpath(str(int(time.time())))
            self.persistence_dir.mkdir(parents=True)
            self.persistence_dir.joinpath("observations").mkdir(parents=True, exist_ok=True)

    def add_datapoint(self, observation: np.ndarray, action: np.ndarray, reward: int):
        """
        Adds the given values as a datapoint to the dataset.

        Args:
            observation: The raw image observation.
            action: The action taken after the observations was recorded.
            reward: The reward at the observation.

        """
        episode = int(len(self.data_points) / self.horizon)
        step = int(len(self.data_points) % self.horizon)
        self.data_points.append(
            DataPoint(step=step, episode=episode, observation=observation, action=action, reward=reward))
        self.same_action_buffer[tuple(action.tolist())].add(len(self.data_points) - 1)

        if self.persistence_dir:
            self.save()

    def save(self):
        data_point = self.data_points[-1]

        path = self.persistence_dir.joinpath("observations", f'{data_point.episode:08d}')
        path.mkdir(exist_ok=True, parents=True)

        cv2.imwrite(os.path.join(path, f'{data_point.step:04d}.png'), cv2.flip(data_point.observation, 0))

        with open(str(self.persistence_dir.joinpath("rewards.yaml")), "a+") as rewards:
            rewards.write(dump([{'e': data_point.episode, 's': data_point.step, 'r': data_point.reward}]))

        with open(str(self.persistence_dir.joinpath("actions.yaml")), "a+") as actions:
            actions.write(
                dump([{'e': data_point.episode, 's': data_point.step, 'a': list(data_point.action.tolist())}]))

    def calculate_same_action_pairs(self):
        self.same_action_pairs = [
            ((i, i + 1), (j, j + 1)) for same_action_set in self.same_action_buffer.values() for i in same_action_set
            for j in same_action_set if
            len(same_action_set) > 1 and i < j and i != len(self.data_points) - 1 and j != len(self.data_points) - 1 and
            self.data_points[i].episode == self.data_points[i + 1].episode and self.data_points[j].episode ==
            self.data_points[j + 1].episode]

    def get_known_action(self):
        index = np.random.randint(0, len(self.data_points))
        action = self.data_points[index].action
        return index, action

    def __len__(self):
        return len(self.same_action_pairs)

    def __getitem__(self, idx):
        same_action_pair = self.same_action_pairs[idx]
        result = {
            'observations': (
                self.transform(self.data_points[same_action_pair[0][0]].observation),
                self.transform(self.data_points[same_action_pair[0][1]].observation),
                self.transform(self.data_points[same_action_pair[1][0]].observation),
                self.transform(self.data_points[same_action_pair[1][1]].observation),
            ),
            'actions': (
                torch.from_numpy(self.data_points[same_action_pair[0][0]].action),
                torch.from_numpy(self.data_points[same_action_pair[0][1]].action),
                torch.from_numpy(self.data_points[same_action_pair[1][0]].action),
                torch.from_numpy(self.data_points[same_action_pair[1][1]].action),
            ),
            'rewards': (
                torch.tensor(self.data_points[same_action_pair[0][0]].reward),
                torch.tensor(self.data_points[same_action_pair[0][1]].reward),
                torch.tensor(self.data_points[same_action_pair[1][0]].reward),
                torch.tensor(self.data_points[same_action_pair[1][1]].reward),
            ),
        }

        return result

    def clear(self):
        self.data_points = []
        self.same_action_buffer = []
