import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    def __init__(self, k: int = 3, distance=euclidean_distance):
        """
        KNN Regressor

        Parameters
        ----------
        k : int
            Number of neighbors to consider.
        distance : callable
            Function to compute the distance between two points.
        """
        self.k = k
        self.distance = distance
        self.train_data = None

    def _fit(self, dataset):
        """
        Store the training data.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model on.
        """
        self.train_data = dataset

    def _get_neighbors(self, sample):
        """
        Get the k-nearest neighbors for a given sample.

        Parameters
        ----------
        sample : np.ndarray
            A single sample (1D array).

        Returns
        -------
        np.ndarray
            The values (y) of the k-nearest neighbors.
        """
        # Garantir que `sample` é 2D para compatibilidade com euclidean_distance
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        # Calcula a distância entre a amostra e todos os pontos do conjunto de treino
        distances = np.array([self.distance(sample, train_sample.reshape(1, -1)) for train_sample in self.train_data.X])

        # Encontra os índices dos k vizinhos mais próximos
        k_indices = np.argsort(distances.flatten())[:self.k]

        # Retorna os valores correspondentes no conjunto de treino
        return self.train_data.y[k_indices]

    def _predict(self, dataset):
        """
        Predict the target values for the given dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset for which to make predictions.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        predictions = np.array([np.mean(self._get_neighbors(sample)) for sample in dataset.X])
        return predictions

    def _score(self, dataset: Dataset) -> float:
        """
        Compute the root mean squared error (RMSE) between predictions and true values.

        Parameters
        ----------
        dataset : Dataset
            The dataset for which to compute the score.

        Returns
        -------
        float
            Root mean squared error (RMSE).
        """
        predictions = self._predict(dataset) 
        return rmse(dataset.y, predictions)  


