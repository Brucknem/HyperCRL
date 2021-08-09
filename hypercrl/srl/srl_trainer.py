import numpy as np
import torch

from hypercrl.srl import SRL


class SRLTrainer:
    def __init__(self, device, horizon):
        self.observation_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.horizon = horizon
        self.device = device

    def add_observation(self, observation: np.ndarray, action: np.ndarray, reward: int):
        # TODO add episodes / boundaries
        self.observation_buffer.append(observation)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        print(len(self.observation_buffer))

    def train(self, srl: SRL):
        print(len(self.observation_buffer))
