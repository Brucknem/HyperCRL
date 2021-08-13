import copy
import time
from typing import Union

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from hypercrl.srl import SRL


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

    def get_similar_states(self, idx):
        """
        Gets the "similar_states" field for a sample.

        Args:
            idx: The id of the datapoint.

        Returns: A dictionary with the observation at idx and the following observation at idx + 1,
                    the action at idx and the reward at idx.
                    If the observations at idx and idx + 1 are from different episodes,
                    the observation at idx is repeated and the action is set to zero.
        """
        data_point: DataPoint = self.data_points[idx]
        action: np.ndarray = data_point.action
        reward = data_point.reward

        if idx != len(self) - 1:
            next_data_point: DataPoint = self.data_points[idx + 1]
        else:
            next_data_point = data_point
            action = np.zeros(action.shape)

        if data_point.episode != next_data_point.episode:
            next_data_point = data_point
            action = np.zeros(action.shape)

        return {
            "observations": [self.transform(data_point.observation), self.transform(next_data_point.observation)],
            "action": torch.from_numpy(action),
            "reward": torch.tensor(reward)
        }

    def get_dissimilar_states(self, idx):
        data_point: DataPoint = self.data_points[idx]
        action: np.ndarray = data_point.action
        reward = data_point.reward

        other_idx = idx
        while abs(other_idx - idx) < 50:
            other_idx = np.random.randint(0, len(self))

        return {
            "observations": [self.transform(data_point.observation),
                             self.transform(self.data_points[other_idx].observation)],
            "action": torch.from_numpy(action),
            "reward": torch.tensor(reward)
        }

    def clear(self):
        self.data_points = []

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "similar_states": self.get_similar_states(idx),
            "dissimilar_states": self.get_dissimilar_states(idx),
        }
        return sample
