from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split

class TestSplit(TestCase):
        def setUp(self):
            self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
            self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        def test_train_test_split_sizes(self):
            train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)
            total_samples = self.dataset.X.shape[0]
            test_samples_size = int(total_samples * 0.2)

            # Verificar o tamanho dos conjuntos
            self.assertEqual(test.X.shape[0], test_samples_size)
            self.assertEqual(train.X.shape[0], total_samples - test_samples_size)

        def test_stratification(self):
            train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)
            
            # Verificar proporções de classes no conjunto original
            unique_classes, original_counts = np.unique(self.dataset.y, return_counts=True)
            original_proportions = original_counts / len(self.dataset.y)

            # Verificar proporções no conjunto de treino e teste
            train_proportions = np.array([np.sum(train.y == cls) for cls in unique_classes]) / len(train.y)
            test_proportions = np.array([np.sum(test.y == cls) for cls in unique_classes]) / len(test.y)

            # Proporções devem ser aproximadamente iguais
            np.testing.assert_almost_equal(train_proportions, original_proportions, decimal=1)
            np.testing.assert_almost_equal(test_proportions, original_proportions, decimal=1)