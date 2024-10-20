from unittest import TestCase
import numpy as np
import os
from si.io.csv_file import read_csv
from si.models.knn_regressor import KNNRegressor
from si.model_selection.split import train_test_split
from si.metrics.rmse import rmse
from datasets import DATASETS_PATH

class TestKNNRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        # Testa o método fit do KNNRegressor
        knn = KNNRegressor(k=3)
        knn.fit(self.dataset)

        # Verifica se os atributos do dataset foram corretamente armazenados
        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2)
        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)

        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])

        # Testa se os valores previstos estão dentro de uma faixa aceitável de valores contínuos (por exemplo, o intervalo dos valores reais)
        self.assertTrue(np.all(predictions >= np.min(test_dataset.y)) and np.all(predictions <= np.max(test_dataset.y)))


    def test_score(self):
        # Testa o método score do KNNRegressor
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2)
        knn.fit(train_dataset)
        score = knn.score(test_dataset)
        self.assertGreaterEqual(score, 0)