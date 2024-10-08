from typing import Tuple
import numpy as np
from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 56) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """

    np.random.seed(random_state)
    permutations = np.random.permutation(dataset.X.shape()[0])
    test_sample_size = int(dataset.shape()[0] * test_size)
    test_idx = permutations [:test_sample_size]
    train_idx = permutations[:test_sample_size:]

    train_dataset = Dataset(dataset.X[train_idx, :], dataset.y[train_idx], features=dataset.features, label=dataset.label)
    test_dataset = Dataset(dataset.X[test_idx, :], dataset.y[test_idx], features=dataset.features, label=dataset.label)

    return train_dataset, test_dataset
