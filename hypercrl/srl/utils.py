import numpy as np
import torch


def scale_to_action_space(action_space, gpuid='cpu', activation='sigmoid'):
    """
    Generates a function that applies a tanh activation to some data and scale the result to the action space
    Args:
        activation: The activation function scaling to [0-1]
        action_space: The (low, high) bounds of the action space
        gpuid: The id of the gpu

    Returns:
        A function that scales the input to the action space

    """
    low = torch.tensor(action_space[0], device=gpuid)
    high = torch.tensor(action_space[1], device=gpuid)

    def inner_scale_to_action_space(x):
        if activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif activation == 'tanh':
            x = torch.tanh(x)
            x = (x + 1.) / 2.
        x = x * (high - low) + low
        return x

    return inner_scale_to_action_space


def calc_mean(means, lengths):
    """
    Calculates the mean of a set of means from different sized samples.

    Args:
        means: The means
        lengths: The original lengths of the samples

    Returns:
        The total mean of the population.

    """
    return np.sum([state * l for state, l in zip(means, lengths)], axis=0) / sum(lengths)


def probabilities_by_size(arr, inverse=True, lower_as_median=False):
    """
    Calculates probabilities proportional to the array values.

    Args:
        arr: The array to sample from. Must be a numeric type
        inverse: Flag to inverse the proportionality of the probabilities.
        lower_as_median: Keeps only the values lower than the median.

    Returns:
        The probabilities proportional to the array values.

    """
    probabilities = np.array(arr)
    if inverse:
        probabilities = -probabilities

    min_probability = min(probabilities)
    if lower_as_median:
        probabilities[probabilities < np.median(probabilities)] = min_probability

    probabilities = probabilities - min_probability
    if sum(probabilities) == 0:
        probabilities = np.array([1] * len(probabilities))

    return probabilities / sum(probabilities)


def sample_by_size(arr, inverse=True, lower_as_median=False):
    """
    Samples the array with probabilities proportional to the array values.

    Args:
        arr: The array to sample from. Must be a numeric type
        inverse: Flag to inverse the proportionality of the probabilities.
        lower_as_median: Keeps only the values lower than the median.

    Returns:
        A sample from the array with probability proportional to the array value.

    """
    p = probabilities_by_size(arr, inverse, lower_as_median)
    return np.random.choice(range(len(arr)), p=p)


def remove_and_move(arr, value):
    """
    Removes the value from the array and decrements all values bigger than the value.
    Args:
        arr: The array to remove from.
        value: The value to remove.

    Returns:
        A copy of the original array without the value.

    """
    result = list(arr)
    if value in arr:
        result.remove(value)
    result = [a if a < value else a - 1 for a in result]
    return result
