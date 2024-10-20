import unittest
import numpy as np
from si.models.lasso_regression import LassoRegression  
from si.metrics import mse  
from si.model_selection.split import train_test_split  

class TestLassoRegression(unittest.TestCase):

    def setUp(self):
        # Simulação de um dataset com 100 exemplos e 3 features
        np.random.seed(42)
        self.X = np.random.rand(100, 3)
        self.y = 3 * self.X[:, 0] + 2 * self.X[:, 1] - 1 * self.X[:, 2] + np.random.normal(0, 0.1, 100)  # Gera y com algum ruído

    def test_fit(self):
        lasso = LassoRegression(l1_penalty=1.0)
        lasso.fit(self.X, self.y)

    
        self.assertIsNotNone(lasso.theta)
        self.assertIsNotNone(lasso.theta_zero)

    def test_predict(self):
        lasso = LassoRegression(l1_penalty=1.0)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        # Treinar o modelo no conjunto de treino
        lasso.fit(X_train, y_train)
        predictions = lasso._predict(X_test)
        # Verificar se o número de predições é igual ao número de exemplos de teste
        self.assertEqual(predictions.shape[0], y_test.shape[0])

    def test_score(self):
        lasso = LassoRegression(l1_penalty=1.0)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        lasso.fit(X_train, y_train)

        score = lasso._score(X_test, y_test)
        # Verificar se o score não é negativo (MSE não pode ser negativo)
        self.assertGreaterEqual(score, 0)
        # Verificar se o score é menor que um valor limiar (considerando o ruído adicionado)
        self.assertLess(score, 1)
