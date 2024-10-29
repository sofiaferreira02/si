import unittest
import numpy as np
from si.data.dataset import Dataset
from si.models.lasso_regression import LassoRegression  
from si.model_selection.split import train_test_split  

class TestLassoRegression(unittest.TestCase):

    def setUp(self):
        # Simulação de um dataset com 100 exemplos e 3 features
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = 3 * X[:, 0] + 2 * X[:, 1] - 1 * X[:, 2] + np.random.normal(0, 0.1, 100)  
        self.dataset = Dataset(X, y)

    def test_fit(self):
        lasso = LassoRegression(l1_penalty=1.0)
        lasso._fit(self.dataset)

        self.assertIsNotNone(lasso.theta)
        self.assertIsNotNone(lasso.theta_zero)

    def test_predict(self):
        lasso = LassoRegression(l1_penalty=1.0)

        train_data, test_data = train_test_split(self.dataset)
        
        lasso._fit(train_data)
        predictions = lasso._predict(test_data)
        # Verificar se o número de predições é igual ao número de exemplos de teste
        self.assertEqual(predictions.shape[0], test_data.y.shape[0])

    def test_score(self):
        lasso = LassoRegression(l1_penalty=1.0)
        train_data, test_data = train_test_split(self.dataset)
        lasso._fit(train_data)

        score = lasso._score(test_data)
        # Verificar se o score não é negativo (MSE não pode ser negativo)
        self.assertGreaterEqual(score, 0)
        # Verificar se o score é menor que um valor limiar (considerando o ruído adicionado)
        self.assertLess(score, 1)