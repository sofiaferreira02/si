from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.lasso_regression import LassoRegression

class TestLassoRegressor(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        model = LassoRegression()
        model._fit(self.train_dataset)

        self.assertIsNotNone(model.theta)
        self.assertEqual(model.theta.size, self.train_dataset.X.shape[1])
        self.assertIsNotNone(model.theta_zero)
        self.assertTrue(hasattr(model, 'cost_history') and len(model.cost_history) > 0)
        self.assertTrue(model.mean is not None and len(model.mean) > 0)
        self.assertTrue(model.std is not None and len(model.std) > 0)

    def test_predict(self):
        model = LassoRegression()
        model._fit(self.train_dataset)
        predictions = model._predict(self.test_dataset)
        self.assertEqual(len(predictions), self.test_dataset.X.shape[0])

    def test_score(self):
        model = LassoRegression(scale=True)
        model._fit(self.train_dataset)
        mse_score = model._score(self.test_dataset)

        # Ajustar tolerância ou valor esperado
        self.assertAlmostEqual(mse_score, 5777.56, delta=0.02)