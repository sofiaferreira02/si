from unittest import TestCase
import numpy as np
from si.statistics.cosine_distance import cosine_distance  
from sklearn.metrics.pairwise import cosine_distances
import os
from si.io.csv_file import read_csv
from datasets import DATASETS_PATH

class TestCosineDistance(TestCase):
    
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_cosine_distance(self):
        """
        Testa a função cosine_distance com a função cosine_distances do scikit-learn.
        Compara os resultados e garante que são semelhantes.
        """
        # Extrai as amostras 
        x = self.dataset.X[0, :]  # Primeira amostra
        y = self.dataset.X[1:, :]  # Restantes

        # Calcula a distância cosseno com a função 
        package_distances = cosine_distance(x, y)

        # Calcula a distância cosseno com a função do scikit-learn
        sklearn_distances = cosine_distances(x.reshape(1, -1), y)

        # Assegura que ambas as distâncias sejam 1D para comparação
        self.assertTrue(np.allclose(package_distances, sklearn_distances.flatten()))


