import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    It computes the euclidean distance of a point (x) to a set of points y.

    Parameters
    ----------
    x: np.ndarray
        Point.
    y: np.ndarray
        Set of points.

    Returns
    -------
    np.ndarray
        Euclidean distance for each point in y.
    """
    return np.sqrt(((x - y) ** 2).sum(axis=1))