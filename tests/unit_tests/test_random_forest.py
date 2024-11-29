import os
from unittest import TestCase
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.random_forest_classifier import RandomForestClassifier
from datasets import DATASETS_PATH

class TestRandomForestClassifier(TestCase):

    def setUp(self):
        self.dataset_path = os.path.join(DATASETS_PATH, "iris", "iris.csv")
        self.dataset = read_csv(filename=self.dataset_path, features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        random_forest = RandomForestClassifier()
        random_forest.fit(self.train_dataset)
        self.assertEqual(random_forest.min_sample_split, 2)
        self.assertEqual(random_forest.max_depth, 10)

    def test_predict(self):
        random_forest = RandomForestClassifier()
        random_forest.fit(self.train_dataset)
        predictions = random_forest.predict(self.test_dataset)
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])

    def test_score(self):
        random_forest = RandomForestClassifier()
        random_forest.fit(self.train_dataset)
        accuracy = random_forest.score(self.test_dataset)
        # Verifica se a accuracy está dentro de um intervalo aceitável (exemplo: > 90%)
        self.assertGreaterEqual(round(accuracy, 2), 0.90)
