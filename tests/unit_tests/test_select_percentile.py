import os
import unittest
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification

class TestSelectPercentile(unittest.TestCase):

    def setUp(self):
        self.dataset_path = os.path.join('datasets', 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.dataset_path, features=True, label=True)

    def test_fit(self):
        
        selector = SelectPercentile(score_func=f_classification, percentile=50)
        selector.fit(self.dataset)

        self.assertTrue(selector.F.shape[0] > 0)
        self.assertTrue(selector.p.shape[0] > 0)


    def test_transform(self):
        selector = SelectPercentile(score_func=f_classification, percentile=50)
        selector.fit(self.dataset)
        transformed_dataset = selector.transform(self.dataset)

        
        self.assertLess(len(transformed_dataset.features), len(self.dataset.features))
        self.assertLess(transformed_dataset.X.shape[1], self.dataset.X.shape[1])
        self.assertEqual(len(transformed_dataset.features),2)
        