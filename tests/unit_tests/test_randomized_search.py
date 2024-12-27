from unittest import TestCase
import os
import numpy as np
from si.io.csv_file import read_csv
from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression
from sklearn.preprocessing import StandardScaler

class TestRandomizedSearchCV(TestCase):

    def setUp(self):
        # Caminho para o dataset
        self.csv_file = os.path.join('datasets', 'breast_bin', 'breast-bin.csv')
        self.dataset = read_csv(filename=self.csv_file, label=True, sep=",")

        # Normalizar os dados
        self.dataset.X = StandardScaler().fit_transform(self.dataset.X)

    def test_randomized_search_cv(self):
        # Inicializar o modelo
        model = LogisticRegression()

        # Definir as distribuições de hiperparâmetros
        parameter_distribution = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.arange(1000, 2000, 200)
        }

        # Realizar Randomized Search CV com 3-fold CV e 10 iterações
        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=parameter_distribution,
            cv=3,
            n_iter=10
        )

        # Validar o número de combinações testadas
        self.assertEqual(len(results['hyperparameters']), 10)
        self.assertEqual(len(results['scores']), 10)

        # Validar se os hiperparâmetros retornados têm os valores esperados
        for param_set in results['hyperparameters']:
            self.assertIn('l2_penalty', param_set)
            self.assertIn('alpha', param_set)
            self.assertIn('max_iter', param_set)

        # Validar que os scores não estão vazios
        self.assertTrue(all(results['scores']))

        # Validar os melhores hiperparâmetros e score
        self.assertIsNotNone(results['best_hyperparameters'])
        self.assertGreater(results['best_score'], -np.inf)
