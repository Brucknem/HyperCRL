import numpy as np
import torch.nn


class Slowness(torch.nn.Module):
    """
    Slowness robotic prior
    """

    def forward(self, state, next_state):
        """

        """
        state_diff = state - next_state
        state_diff_norm = state_diff.norm(2, dim=1)
        return (state_diff_norm ** 2).mean()


class Variability(torch.nn.Module):
    """
    Variability robotic prior
    """

    def forward(self, state, other_state):
        state_diff = state - other_state
        state_diff_norm = state_diff.norm(2, dim=1)
        exp = torch.exp(-state_diff_norm)
        return exp.mean()


class Proportionality(torch.nn.Module):

    def forward(self, state, next_state, other_state, next_other_state):
        state_diff = state - next_state
        other_state_diff = other_state - next_other_state

        state_diff_norm = state_diff.norm(2, dim=1)
        other_state_diff_norm = other_state_diff.norm(2, dim=1)
        return 0
