import numpy as np

from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor:
    """
    KNN Regressor
    The k-Nearst Neighbors regressor is a machine learning model that estimates the mean of the k-nearest samples in
    the training data, based on a similarity measure (e.g., distance functions).

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

    def __init__(self, k: int, distance: euclidean_distance = euclidean_distance):
        """
        Construtor da class KNNRegressor.

        :param k: Número de k exemplos de nearest neighbors a considerar.
        :param distance: Função que calcula a distância entre a amostra e as amostras do dataset de treino
        """
        # Parâmetros
        self.k = k
        self.distance = distance

        # Atributos
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Método para fazer o fit do modelo de acordo com o input dataset.

        :param dataset: Dataset de treino.

        :return: self. O modelo treinado
        """
        self.dataset = dataset
        return self

    def _get_closet_label_mean(self, x: np.ndarray) -> np.ndarray:
        """
        Método que calcula a média das labels mais próximas de uma dada sample - highest frequency classes.

        :param x: Array de samples.

        :return: Array com índices das médias das labels.
        """
        # Compute distance between the samples and the dataset
        distances = self.distance(x, self.dataset.X)

        # Sort the distances and get indexes of nearest neighbors
        knn = np.argsort(distances)[:self.k]  # get the first k indexes of the sorted distances array

        # Get the labels of the obtained indexes
        knn_labels = self.dataset.y[knn]

        # Computes the mean of the matching classes
        knn_labels_mean = np.mean(knn_labels)

        return knn_labels_mean

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Método para prever as médias das labels de um dado dataset.

        :param dataset: Dataset

        :return: Array com as previsões do modelo.
        """
        return np.apply_along_axis(self._get_closet_label_mean, axis=1, arr=dataset.X)  # axis=1 por ser nas linhas

    def score(self, dataset: Dataset) -> float:
        """
        Método que calcula o score do modelo - o erro entre os valores estimados e reais, usando a fórmula RMSE.

        :param dataset: Dataset

        :return: Valor RMSE do modelo.
        """
        predictions = self.predict(dataset)

        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset=dataset_, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNRegressor(k=3, distance=euclidean_distance)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The RMSE value of the model is: {score}')