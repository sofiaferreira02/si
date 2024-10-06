import os
from unittest import TestCase

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA


class TestPCA(TestCase):

    def setUp(self):
        # Caminho para o dataset iris
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        # Carregar o dataset
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        # Criar o PCA com 2 componentes
        estimator = PCA(n_components=2).fit(self.dataset)

        # Testar o número de componentes
        self.assertEqual(estimator.n_components, 2)
        self.assertEqual(estimator.components_.shape[0], 2)  # Deve ter 2 componentes principais

    def test_transform(self):
        # Aplicar PCA
        estimator = PCA(n_components=2).fit(self.dataset)
        new_dataset = estimator.transform(self.dataset)

        # Verificar se o número de dimensões foi reduzido
        self.assertEqual(new_dataset.X.shape[1], 2)
        self.assertEqual(new_dataset.X.shape[0], self.dataset.X.shape[0])