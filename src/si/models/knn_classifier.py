import numpy as np
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from typing import Callable, Union
from si.metrics.accuracy import accuracy

class KNNClassifier:
    """
    KNN Classifier
    The k-Nearst Neighbors classifier is a machine learning model that classifies new samples based on
    a similarity measure (e.g., distance functions). This algorithm predicts the classes of new samples by
    looking at the classes of the k-nearest samples in the training data.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN classifier

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNClassifier
            The fitted model
        """
        self.dataset = dataset
        return self


    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of

        Returns
        -------
        label: str or int
            The closest label
        """
        distances = self.distance(sample, self.dataset.X)
        closest_neighbors_idx = np.argsort(distances)[:self.k]
        closest_labels  = self.dataset.y[closest_neighbors_idx]
        labels, counts = np.unique(closest_labels, return_counts=True)
        return labels[np.argmax(counts)]
    

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)