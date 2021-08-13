import numpy as np
import torch.nn


class SlownessPrior(torch.nn.Module):
    """
    The slowness robotic prior: https://arxiv.org/pdf/1802.04181.pdf Eq: (11)

    Two states close to each other in time are also close to each other in the state representation space.
    """

    def forward(self, state: torch.Tensor, next_state: torch.Tensor):
        """
        Calculates the forward pass for the robotic prior.

        The states have to be near in the time domain, ideally directly following states.

        Args:
            state: Some state.
            next_state: The following state.

        Returns:
            The value of the robotic prior.

        """
        state_diff = state - next_state
        state_diff_norm = state_diff.norm(2, dim=1)

        result = state_diff_norm ** 2
        return result.mean()


class VariabilityPrior(torch.nn.Module):
    """
    The variability robotic prior: https://arxiv.org/pdf/1802.04181.pdf Eq: (12)

    Positions of relevant objects vary, and learning state representations should then focus on moving objects.
    """

    def forward(self, state: torch.Tensor, other_state: torch.Tensor):
        """
        Calculates the forward pass for the robotic prior.

        The states should differ in time.

        Args:
            state: Some state.
            other_state: Another state.

        Returns:
            The value of the robotic prior.

        """
        state_diff = state - other_state
        state_diff_norm = state_diff.norm(2, dim=1)

        result = torch.exp(-state_diff_norm)
        return result.mean()


class ProportionalityPrior(torch.nn.Module):
    """
    The proportionality robotic prior: https://arxiv.org/pdf/1802.04181.pdf Eq: (13)

    Two identical actions should result in two proportional magnitude state variations.
    """

    def forward(self, state: torch.Tensor, next_state: torch.Tensor, other_state: torch.Tensor,
                other_next_state: torch.Tensor):
        """
        Calculates the forward pass for the robotic prior.

        The state pairs have to be directly following in time.
        The actions applied at the states have to be equal.

        Args:
            state: Some state.
            next_state: The following state.
            other_state: Another state.
            other_next_state: The following state for the other state.

        Returns:
            The value of the robotic prior.

        """
        state_diff = state - next_state
        other_state_diff = other_state - other_next_state

        norm_diff = other_state_diff.norm(2, dim=1) - state_diff.norm(2, dim=1)

        result = norm_diff ** 2
        return result.mean()


class RepeatabilityPrior(torch.nn.Module):
    """
    The repeatability robotic prior: https://arxiv.org/pdf/1802.04181.pdf Eq: (14)

    Two identical actions applied at similar states should provide similar state variations,
    not only in magnitude but also in direction.
    """

    def forward(self, state: torch.Tensor, next_state: torch.Tensor, other_state: torch.Tensor,
                other_next_state: torch.Tensor):
        """
        Calculates the forward pass for the robotic prior.

        The state pairs have to be directly following in time.
        The actions applied at the states have to be equal.

        Args:
            state: Some state.
            next_state: The following state.
            other_state: Another state.
            other_next_state: The following state for the other state.

        Returns:
            The value of the robotic prior.

        """
        exp_diff = other_state - state
        exp_diff_norm = exp_diff.norm(2, dim=1)

        delta_diff = (other_state - other_next_state) - (state - next_state)
        delta_diff_norm = delta_diff.norm(2, dim=1)

        result = torch.exp(-(exp_diff_norm ** 2)) * delta_diff_norm ** 2
        return result.mean()


class CausalityPrior(torch.nn.Module):
    """
    The causality robotic prior: https://arxiv.org/pdf/1709.05185.pdf Eq: (4)

    If two states on which the same action is applied give two different rewards, they should not be close
    to each other in the state representation space.
    """

    def forward(self, state: torch.Tensor, other_state: torch.Tensor):
        """
        Calculates the forward pass for the robotic prior.

        The states should differ in time.
        The actions applied at the states have to be equal.
        The rewards at the following states have to be different.

        Args:
            state: Some state.
            other_state: Another state.

        Returns:
            The value of the robotic prior.

        """
        state_diff = other_state - state
        state_diff_norm = state_diff.norm(2, dim=1)

        result = torch.exp(-(state_diff_norm ** 2))
        return result.mean()


class ReferencePointPrior(torch.nn.Module):
    """
    The reference point robotic prior: https://arxiv.org/pdf/1709.05185.pdf Eq: (5)

    Two states corresponding to the same reference point should be close to each other.
    """

    def forward(self, state: torch.Tensor, other_state: torch.Tensor):
        """
        Calculates the forward pass for the robotic prior.

        The states should differ in time.
        The original states resulting in the states after representation encoding should be
        equal to some reference state.

        Args:
            state: Some state.
            other_state: Another state.

        Returns:
            The value of the robotic prior.

        """
        state_diff = other_state - state
        state_diff_norm = state_diff.norm(2, dim=1)

        result = state_diff_norm ** 2
        return result.mean()
