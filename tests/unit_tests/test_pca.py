import os
from unittest import TestCase
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA
import numpy as np

class TestPCA(TestCase):
    def setUp(self):
        self.csv_file = os.path.join('datasets', 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        # Criar o PCA com 2 componentes
        estimator = PCA(n_components=2)
        estimator._fit(self.dataset)  

        # Testar se o modelo foi treinado corretamente
        self.assertEqual(estimator.n_components, 2)
        self.assertEqual(estimator.components.shape[0], 2)  # Deve ter 2 componentes principais
        self.assertEqual(estimator.components.shape[1], self.dataset.X.shape[1])  # Deve ser igual ao número de features originais
        self.assertEqual(len(estimator.explained_variance), 2)  # Deve ter 2 variâncias explicadas
        self.assertTrue(estimator.fitted)  # O modelo deve estar marcado como "fitted"

    def test_transform(self):
        # Aplicar PCA
        estimator = PCA(n_components=2)
        estimator._fit(self.dataset)
        new_dataset = estimator._transform(self.dataset)

        # Verificar se o número de dimensões foi reduzido
        self.assertEqual(new_dataset.X.shape[1], 2)  # Deve reduzir para 2 dimensões
        self.assertEqual(new_dataset.X.shape[0], self.dataset.X.shape[0])  # O número de amostras deve permanecer o mesmo

    def test_mean_centering(self):
        # Testar se os dados foram centralizados corretamente
        estimator = PCA(n_components=2)
        estimator._fit(self.dataset)
        centered_data = self.dataset.X - estimator.mean
        self.assertTrue(np.allclose(centered_data.mean(axis=0), 0, atol=1e-8))  # Após centralizar, a média deve ser próxima de 0
