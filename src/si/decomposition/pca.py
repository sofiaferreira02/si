import numpy as np

from si.data.dataset import Dataset


class PCA:
    """
    PCA clustering usando a técnica de álgebra linear SVD (Singular Value Decomposition).
    Transforma variáveis, possivelmente correlacionadas, num nº menor de variáveis que sejam capazes de representar os
    dados.
    Objetivo: Reduzir o nº de dimensões de um conjunto de dados para facilitar a visualização, análise e interpretação.

    Parâmetros
    ----------
    n_components: int
        Número de componentes.

    Atributos
    ----------
    mean:
        Média das amostras
    components:
        Componentes principais aka matriz unitária dos eigenvectors
    explained_variance:
        Variância explicada aka matriz diagonal dos eigenvalues
    """

    def __init__(self, n_components: int):
        """
        Construtor do PCA, usando a técnica de álgebra linear SVD (Singular Value Decomposition).

        :param n_components: Número de componentes a considerar para a análise.
        """
        # Parâmetros:
        self.n_components = n_components

        # Atributos:
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _get_centered_data(self, dataset: Dataset) -> np.ndarray:
        """
        Método para centrar os dados.

        :param dataset: Dataset.
        :return: ndarray com os dados centrados.
        """
        # Calculate centered data
        self.mean = np.mean(dataset.X, axis=0) # axis=0 - coluna
        self.centered_data = dataset.X - dataset.X.mean(axis=0, keepdims=True)
        return self.centered_data

    def _get_components(self, dataset: Dataset) -> np.ndarray:
        """
        Método que calcula os componentes do dataset.

        :param dataset: Dataset.

        :return: ndarray com os componentes.
        """
        # Get centered data
        self.centered_data = self._get_centered_data(dataset)

        # Calculate SVD
        self.u, self.s, self.vt = np.linalg.svd(self.centered_data, full_matrices=False)

        # Componentes principais
        self.components = self.vt[:self.n_components] # colunas com os primeiros n_components

        return self.components

    def _get_explained_variance(self, dataset: Dataset) -> np.ndarray:
        """
        Método que calcula a explained variance.

        :param dataset: Dataset.

        :return: ndarray (vetor) com a explained variance.
        """
        # Calculate explained variance (ev)
        len_dataset = len(dataset.X)
        ev = (self.s ** 2) / (len_dataset - 1)
        explained_variance = ev[:self.n_components]

        return explained_variance

    def fit(self, dataset: Dataset) -> "PCA":
        """
        Método que recebe o dataset, faz fit dos dados e guarda os valores da média, os componentes principais e a
        variância explicada de cada observação,

        :param dataset: Dataset, input dataset

        :return: self
        """
        self.components = self._get_components(dataset)
        self.explained_variance = self._get_explained_variance(dataset)

        return self

    def transform(self, dataset: Dataset) -> np.ndarray:
        """
        Método que transforma os dados ao calcular o Singular Value Decomposition (SVD)

        :param dataset: Dataset, input dataset

        :return: Dataset transformado
        """
        # Get matriz V transposta
        v = self.vt.T

        # Get dados transformados
        X_reduced = np.dot(self.centered_data, v)
        # return Dataset(X=transformed_data, y=dataset.y, features_names=dataset.features_names, label_names=dataset.label_names)
        return X_reduced

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        """
        Método que faz o fit e transforma o dataset.

        :param dataset: Dataset, input dataset

        :return: Dataset
        """
        self.fit(dataset)
        return self.transform(dataset=dataset)


if __name__ == "__main__":
    import pandas as pd
    dataset_: Dataset = Dataset.from_random(10, 5)

    df = pd.DataFrame(data=np.column_stack((dataset_.X, dataset_.y)), columns=dataset_.features + [dataset_.label])

    print("Dataframe random:\n", dataset_.from_dataframe(df))
    dataset_pca = PCA(n_components=2)
    print("Pós PCA:\n", dataset_pca.fit_transform(dataset=dataset_))