import numpy as np
import torch.nn


class SlownessPrior(torch.nn.Module):
    """
    The slowness robotic prior: https://arxiv.org/pdf/1802.04181.pdf Eq: (11)
    """

    def forward(self, state: torch.Tensor, next_state: torch.Tensor):
        """

        """
        state_diff = state - next_state
        state_diff_norm = state_diff.norm(2, dim=1)

        result = (state_diff_norm ** 2)
        return result.mean()


class VariabilityPrior(torch.nn.Module):
    """
    The variability robotic prior: https://arxiv.org/pdf/1802.04181.pdf Eq: (12)
    """

    def forward(self, state: torch.Tensor, other_state: torch.Tensor):
        state_diff = state - other_state
        state_diff_norm = state_diff.norm(2, dim=1)

        result = torch.exp(-state_diff_norm)
        return result.mean()


class ProportionalityPrior(torch.nn.Module):
    """
    The proportionality robotic prior: https://arxiv.org/pdf/1802.04181.pdf Eq: (13)
    """

    def forward(self, state: torch.Tensor, next_state: torch.Tensor, other_state: torch.Tensor,
                other_next_state: torch.Tensor):
        state_diff = state - next_state
        other_state_diff = other_state - other_next_state

        norm_diff = other_state_diff.norm(2, dim=1) - state_diff.norm(2, dim=1)

        result = norm_diff ** 2
        return result.mean()


class RepeatabilityPrior(torch.nn.Module):
    """
    The repeatability robotic prior: https://arxiv.org/pdf/1802.04181.pdf Eq: (14)
    """

    def forward(self, state: torch.Tensor, next_state: torch.Tensor, other_state: torch.Tensor,
                other_next_state: torch.Tensor):
        exp_diff = other_state - state
        exp_diff_norm = exp_diff.norm(2, dim=1)

        delta_diff = (other_state - other_next_state) - (state - next_state)
        delta_diff_norm = delta_diff.norm(2, dim=1)

        result = torch.exp(-(exp_diff_norm ** 2)) * delta_diff_norm ** 2
        return result.mean()


class CausalityPrior(torch.nn.Module):
    """
    The causality robotic prior: https://arxiv.org/pdf/1709.05185.pdf Eq: (4)
    """

    def forward(self, state: torch.Tensor, other_state: torch.Tensor):
        state_diff = other_state - state
        state_diff_norm = state_diff.norm(2, dim=1)

        result = torch.exp(-(state_diff_norm ** 2))
        return result.mean()


class ReferencePointPrior(torch.nn.Module):
    """
    The reference point robotic prior: https://arxiv.org/pdf/1709.05185.pdf Eq: (5)
    """

    def forward(self, state: torch.Tensor, other_state: torch.Tensor):
        state_diff = other_state - state
        state_diff_norm = state_diff.norm(2, dim=1)

        result = state_diff_norm ** 2
        return result.mean()
