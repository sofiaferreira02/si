import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset


class PCA(Transformer):
    def __init__(self, n_components: int, **kwargs):
        """
        Principal Component Analysis (PCA)

        Parameters
        ----------
        n_components : int
            Number of principal components to retain.
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.fitted = False
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset) -> "PCA":
        """
        Fit the PCA model to the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the PCA model to.

        Returns
        -------
        PCA
            Fitted PCA instance.

        Raises
        ------
        ValueError
            If the number of components is invalid.
        """
        # Validate the number of components
        if not 0 < self.n_components <= dataset.shape()[1]:
            raise ValueError("n_components must be a positive integer no greater than the number of features.")

        # Center the dataset
        self.mean = np.mean(dataset.X, axis=0)
        centered_data = dataset.X - self.mean

        # Calculate covariance matrix and perform eigen decomposition
        cov_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Select the top n_components based on eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.components = eigenvectors[:, sorted_indices].T
        self.explained_variance = eigenvalues[sorted_indices] / np.sum(eigenvalues)

        self.fitted = True
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transforma o dataset usando os componentes principais ajustados.

        Parameters
        ----------
        dataset : Dataset
            O dataset a ser transformado.

        Returns
        -------
        Dataset
            Dataset transformado com dimensões reduzidas.
        """
        if not self.fitted:
            raise ValueError("O modelo PCA deve ser ajustado antes de chamar _transform.")

        # Centralizar os dados
        centered_data = dataset.X - self.mean

        # Projetar os dados nos componentes principais
        reduced_data = np.dot(centered_data, self.components.T)

        # Criar um novo dataset com as features reduzidas
        feature_names = [f"PC{i+1}" for i in range(self.n_components)]
        return Dataset(X=reduced_data, y=dataset.y, features=feature_names, label=dataset.label)