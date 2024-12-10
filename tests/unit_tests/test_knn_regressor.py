from unittest import TestCase
import numpy as np
import os
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance
from si.models.knn_regressor import KNNRegressor  
from datasets import DATASETS_PATH

class TestKNNRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNRegressor(k=3)
        knn._fit(self.dataset)
        # Verifica se o dataset foi armazenado corretamente no modelo
        self.assertTrue(knn.train_data is not None)
        self.assertTrue(np.all(self.dataset.X == knn.train_data.X))
        self.assertTrue(np.all(self.dataset.y == knn.train_data.y))

    def test_get_neighbors(self):
        knn = KNNRegressor(k=2, distance=euclidean_distance)
        knn._fit(self.dataset)
        sample = self.dataset.X[0]
        neighbors = knn._get_neighbors(sample)
        # Verifica se os valores dos vizinhos estão no conjunto de treino
        for neighbor_value in neighbors:
            self.assertIn(neighbor_value, self.dataset.y)

    def test_predict(self):
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)
        knn._fit(train_dataset)
        predictions = knn._predict(test_dataset)
        # Verifica o tamanho das previsões
        self.assertEqual(predictions.shape[0], test_dataset.X.shape[0])
        # Verifica se as previsões estão aceitáveis
        self.assertTrue(np.all(predictions >= np.min(train_dataset.y)))
        self.assertTrue(np.all(predictions <= np.max(train_dataset.y)))

    def test_score(self):
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)
        knn._fit(train_dataset)
        score = knn._score(test_dataset)
        # Verifica se o score (RMSE) é válido (deve ser >= 0)
        self.assertGreaterEqual(score, 0)

        # Calcula as previsões e verifica o score
        predictions = knn._predict(test_dataset)
        expected_rmse = rmse(test_dataset.y, predictions)
        self.assertAlmostEqual(score, expected_rmse, places=5)
