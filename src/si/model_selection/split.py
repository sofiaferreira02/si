from typing import Tuple
import numpy as np
from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
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
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and testing sets while preserving the class distribution in each subset.

    Parameters
    ----------
    dataset: Dataset
        - Dataset object to split
    test_size: float
        - Size of the test set. By default, 20%
    random_state: int
        - Random seed for reproducibility
    
    Returns
    -------
    Tuple[Dataset, Dataset]
        - A tuple where the first element is the training dataset and the second element is the testing dataset

    Raises
    -------
    ValueError
        - If test_size is not a float between 0 and 1
    """
    if not (0 < test_size < 1):
        raise ValueError("O parâmetro test_size deve ser um valor entre 0 e 1.")
    
    rng = np.random.default_rng(random_state)
    unique_classes, class_counts = np.unique(dataset.y, return_counts=True)
    
    train_indices = []
    test_indices = []

    for cls, count in zip(unique_classes, class_counts): # Estratificar com base nas classes
        class_indices = np.where(dataset.y == cls)[0]
        rng.shuffle(class_indices)
        
        test_count = int(count * test_size)
        test_indices.extend(class_indices[:test_count])
        train_indices.extend(class_indices[test_count:])
    
        train_data = Dataset(
        X=dataset.X[train_indices], 
        y=dataset.y[train_indices], 
        features=dataset.features, 
        label=dataset.label
    )
    test_data = Dataset(
        X=dataset.X[test_indices], 
        y=dataset.y[test_indices], 
        features=dataset.features, 
        label=dataset.label
    )
    
    return train_data, test_data
