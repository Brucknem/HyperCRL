import copy
import math
import sys
import time
from typing import Union

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from hypercrl.srl import SRL
from collections import defaultdict


class DataPoint:
    """
    Wrapper for a datapoint within an episode.
    """

    def __init__(self, episode: int, observation: np.ndarray, action: np.ndarray, reward: int):
        """
        Args:
            episode: The id of the episode.
            observation: The raw image observation.
            action: The action taken after the observations was recorded.
            reward: The reward at the observation.
        """
        self.episode = episode
        self.observation = observation
        self.action = action
        self.reward = reward


class SRLDataSet(Dataset):
    """
    The state representation learning dataset.
    """

    def __init__(self, transform: Union[any, str] = 'default', seed: int = 12345):
        """
        Args:
            transform: A transformation that is applied to the raw image observation.
                        If set to 'default' subtract mean and divide by variance of ImageNet dataset.
            seed: The random seed for numpy.
        """
        self.data_points = []

        self.seed = seed
        np.random.seed(self.seed)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if transform is 'default':
            transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if transform is not None:
            self.transform = transforms.Compose([self.transform, transform])

        self.same_action_buffer = defaultdict(set)
        self.same_action_pairs = []

    def add_datapoint(self, episode: int, observation: np.ndarray, action: np.ndarray, reward: int):
        """
        Adds the given values as a datapoint to the dataset.

        Args:
            episode: The id of the episode.
            observation: The raw image observation.
            action: The action taken after the observations was recorded.
            reward: The reward at the observation.

        """
        self.data_points.append(DataPoint(episode=episode, observation=observation, action=action, reward=reward))
        self.same_action_buffer[tuple(action)].add(len(self.data_points) - 1)

    def calculate_same_action_pairs(self):
        self.same_action_pairs = [
            ((i, i + 1), (j, j + 1)) for same_action_set in self.same_action_buffer.values() for i in same_action_set
            for j in same_action_set if
            len(same_action_set) > 1 and i < j and i != len(self.data_points) - 1 and j != len(
                self.data_points) - 1 and self.data_points[i].episode == self.data_points[i + 1].episode and
            self.data_points[j].episode == self.data_points[j + 1].episode]

    # def calculate_same_action_pairs(self):
    #     for i in range(len(self.data_points) - 1):
    #         for j in range(len(self.data_points) - 1):
    #             if i >= j:
    #                 continue
    #
    #             entry = self.data_points[i]
    #             other = self.data_points[j]
    #             if (entry.action - other.action).sum() != 0:
    #                 continue
    #
    #             if entry.episode != self.data_points[i + 1].episode:
    #                 continue
    #
    #             if other.episode != self.data_points[j + 1].episode:
    #                 continue
    #
    #             self.same_action_pairs.append(((i, i + 1), (j, j + 1)))

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
                (self.transform(self.data_points[same_action_pair[0][0]].observation),
                 self.transform(self.data_points[same_action_pair[0][1]].observation)),
                (self.transform(self.data_points[same_action_pair[1][0]].observation),
                 self.transform(self.data_points[same_action_pair[1][1]].observation)),
            ),
            'actions': (
                (torch.from_numpy(self.data_points[same_action_pair[0][0]].action),
                 torch.from_numpy(self.data_points[same_action_pair[0][1]].action)),
                (torch.from_numpy(self.data_points[same_action_pair[1][0]].action),
                 torch.from_numpy(self.data_points[same_action_pair[1][1]].action)),
            ),
            'rewards': (
                (torch.tensor(self.data_points[same_action_pair[0][0]].reward),
                 torch.tensor(self.data_points[same_action_pair[0][1]].reward)),
                (torch.tensor(self.data_points[same_action_pair[1][0]].reward),
                 torch.tensor(self.data_points[same_action_pair[1][1]].reward)),
            ),
        }

        return result

    def clear(self):
        self.data_points = []
