import os
from unittest import TestCase
from datasets import DATASETS_PATH
import numpy as np

import os
from si.io.csv_file import read_csv

from si.statistics.sigmoid_function import sigmoid_function

class TestSigmoidFunction(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_sigmoid_function(self):

        x = np.array([1,2,3])

        self.assertTrue(all(sigmoid_function(x)> 0))
        self.assertTrue(all(sigmoid_function(x)< 1))
