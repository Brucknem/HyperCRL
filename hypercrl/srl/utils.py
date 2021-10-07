import numpy as np


def probabilities_by_size(arr, inverse=True, lower_as_median=False):
    """
    Calculates probabilities proportional to the array values.

    Args:
        arr: The array to sample from. Must be a numeric type
        inverse: Flag to inverse the proportionality of the probabilities.
        lower_as_median: Keeps only the values lower than the median.

    Returns:

    """
    probabilities = np.array(arr)
    if inverse:
        probabilities = -probabilities

    min_probability = min(probabilities)
    if lower_as_median:
        probabilities[probabilities < np.median(probabilities)] = min_probability

    probabilities = probabilities - min_probability
    return probabilities / sum(probabilities)


def sample_by_size(arr, inverse=True, lower_as_median=False):
    """
    Samples the array with probabilities proportional to the array values.

    Args:
        arr: The array to sample from. Must be a numeric type
        inverse: Flag to inverse the proportionality of the probabilities.
        lower_as_median: Keeps only the values lower than the median.

    Returns:

    """
    p = probabilities_by_size(arr, inverse, lower_as_median)
    return np.random.choice(range(len(arr)), p=p)
