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
        x = np.array([1, 2, 3])  # Vetor único
        y = np.array([[1, 2, 3], [4, 5, 6]])  # Conjunto de vetores

        distance = cosine_distance(x, y)

        # Calcula as distâncias usando o scikit-learn
        sklearn_distance = cosine_distances(x.reshape(1, -1), y)

        # Compara os resultados
        self.assertTrue(np.allclose(distance, sklearn_distance.flatten()))



