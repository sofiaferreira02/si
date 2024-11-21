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

        # Ler e preparar o dataset
        dataset = read_csv(self.csv_file, sep=",", label=True)
        dataset.X = StandardScaler().fit_transform(dataset.X)
        self.dataset = dataset

    def test_randomized_search_cv(self):
        # Inicializar o modelo
        model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

        # Definir as distribuições de parâmetros
        parameter_distribution = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.linspace(1000, 2000, 200)
        }

        # Realizar Randomized Search CV com 3-fold CV e 10 iterações
        results = randomized_search_cv(model=model,
                                       dataset=self.dataset,
                                       parameter_distribution=parameter_distribution,
                                       cv=3,
                                       n_iter=10)

        # Validar o número de combinações testadas
        self.assertEqual(len(results['parameters']), 10)
        self.assertEqual(len(results['train']), 10)
        self.assertEqual(len(results['test']), 10)

        # Validar se os parâmetros retornados têm os valores esperados
        for param_set in results['parameters']:
            self.assertIn('l2_penalty', param_set)
            self.assertIn('alpha', param_set)
            self.assertIn('max_iter', param_set)

        # Validar os valores de seed são únicos
        self.assertEqual(len(set(results['seed'])), 10)

        # Validar que os scores não estão vazios
        self.assertTrue(all(results['train']))
        self.assertTrue(all(results['test']))