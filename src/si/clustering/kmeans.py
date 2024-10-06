from typing import Callable
import numpy as np
from si.data.dataset import Dataset
from si.base.transformer import Transformer
from si.base.model import Model
from src.si.statistics.euclidean_distance import euclidean_distance


class KMeans(Transformer, Model):
    """
    It performs k-means clustering on the dataset.
    It groups samples into k clusters by trying to minimize the distance between samples and their closest centroid.
    It returns the centroids and the indexes of the closest centroid for each point.

    Parameters
    ----------
    k: int
        Number of clusters.
    max_iter: int
        Maximum number of iterations.
    distance: Callable
        Distance function.

    Attributes
    ----------
    centroids: np.array
        Centroids of the clusters.
    labels: np.array
        Labels of the clusters.
    """

    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = euclidean_distance, **kwargs):
        """
        K-means clustering algorithm.

        Parameters
        ----------
        k: int
            Number of clusters.
        max_iter: int
            Maximum number of iterations.
        distance: Callable
            Distance function.
        """
        # parameters
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        # attributes
        self.centroids = None
        self.labels = None

    def _init_centroids(self, dataset: Dataset):
        """
        It generates initial k centroids.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        seeds = np.random.permutation(dataset.shape()[0])[:self.k]
        self.centroids = dataset.X[seeds, :]

    def _get_closest_centroid(self, sample: np.ndarray) ->int:
        """
        Get the closest centroid to each data point.

        Parameters
        ----------
        sample : np.ndarray, shape=(n_features,)
            A sample.

        Returns
        -------
        np.ndarray
            The closest centroid to each data point.
        """
        distance_ = self.distance(sample, self.centroids)
        return np.argmin(distance_)
        

    def fit(self, dataset: Dataset) -> 'KMeans':
        """
        It fits k-means clustering on the dataset.
        The k-means algorithm initializes the centroids and then iteratively updates them until convergence or max_iter.
        Convergence is reached when the centroids do not change anymore.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        KMeans
            KMeans object.
        """
        # generate initial centroids
        self._init_centroids(dataset = dataset)
    

        # get closest centroid
        new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)

        convergence = False
        j = 0
        while not convergence and j < self.max_iter:

            new_centroids = []
            for i in range(self.k):
                new_centroid = np.mean(dataset.X[new_labels == i], axis=0)
                new_centroids.append(new_centroid)

            self.centroids = np.array(new_centroids)

            convergence = np.any(new_labels != self.labels)
            convergence = np.all(new_labels == self.labels)
            j = 1
            self.labels = new_labels

        return self

    def _get_distances(self, sample: np.ndarray) -> np.ndarray:
        """
        It computes the distance between each sample and the closest centroid.

        Parameters
        ----------
        sample : np.ndarray, shape=(n_features,)
            A sample.

        Returns
        -------
        np.ndarray
            Distances between each sample and the closest centroid.
        """
        return self.distance(sample, self.centroids)

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        It transforms the dataset.
        It computes the distance between each sample and the closest centroid.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        np.ndarray
            Transformed dataset.
        """
        centroids_distances = np.apply_along_axis(self._get_distances, axis=1, arr=dataset.X)
        return centroids_distances

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        It fits and transforms the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        np.ndarray
            Transformed dataset.
        """
        self.fit(dataset)
        return self.transform(dataset)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the labels of the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        return np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)

    def fit_predict(self, dataset: Dataset) -> np.ndarray:
        """
        It fits and predicts the labels of the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        self.fit(dataset)
        return self.predict(dataset)
