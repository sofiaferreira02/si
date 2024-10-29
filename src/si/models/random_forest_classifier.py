from typing import Literal

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:
    """
       A class representing a Random Forest classifier.

       Parameters: n_estimators (int): The number of decision trees to use in the ensemble. max_features (int): The
       maximum number of features to use per tree. If None, it defaults to sqrt(n_features). min_sample_split (int):
       The minimum number of samples required to split an internal node. max_depth (int): The maximum depth of the
       decision trees in the ensemble. mode (Literal['gini', 'entropy']): The impurity calculation mode for
       information gain (either 'gini' or 'entropy'). seed (int): The random seed to ensure reproducibility.

       Estimated Parameters:
           trees (list of tuples): List of decision trees and their respective features used for training.

       Methods:
           - fit(dataset: Dataset) -> RandomForestClassifier:
               Fits the Random Forest classifier to a given dataset.

           - predict(dataset: Dataset) -> np.ndarray:
               Predicts labels for a given dataset using the ensemble of decision trees.

           - score(dataset: Dataset) -> float:
               Computes the accuracy of the model's predictions on a dataset.
       """

    def __init__(self, n_estimators: int = 100, max_features: int = None, min_sample_split: int = 2,
                 max_depth: int = 10, mode: Literal['gini', 'entropy'] = 'gini', seed: int = None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
                Fits the Random Forest classifier to a given dataset.

                Parameters:
                    dataset (Dataset): The dataset to fit the model to.

                Returns:
                    RandomForestClassifier: The fitted model.
                """
        if self.seed is not None:
            np.random.seed(self.seed)
        n_samples, n_features = dataset.shape()
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        for x in range(self.n_estimators):
            bootstrap_samples = np.random.choice(n_samples, n_samples,
                                                 replace=True)
            bootstrap_features = np.random.choice(n_features, self.max_features,
                                                  replace=False)
            random_dataset = Dataset(dataset.X[bootstrap_samples][:, bootstrap_features], dataset.y[bootstrap_samples])

            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split, max_depth=self.max_depth,
                                          mode=self.mode)
            tree.fit(random_dataset)
            self.trees.append((bootstrap_features, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
                Predicts labels for a given dataset using the ensemble of decision trees.

                Parameters:
                    dataset (Dataset): The dataset to make predictions for.

                Returns:
                    np.ndarray: The predicted labels.

                Note: The predictions are obtained by aggregating the predictions of individual decision trees in the
                ensemble.
        """
        n_samples = dataset.shape()[0]
        predictions = np.zeros((self.n_estimators, n_samples), dtype=object)

        # for each tree
        row = 0
        for features, tree in self.trees:
            tree_preds = tree.predict(dataset)
            predictions[row, :] = tree_preds
            row += 1

        def majority_vote(sample_predictions):
            unique_classes, counts = np.unique(sample_predictions, return_counts=True)
            most_common = unique_classes[np.argmax(counts)]
            return most_common

        majority_prediction = np.apply_along_axis(majority_vote, axis=0, arr=predictions)

        return majority_prediction

    def _score(self, dataset: Dataset) -> float:
        """
                Computes the accuracy of the model's predictions on a dataset.

                Parameters:
                    dataset (Dataset): The dataset to calculate the accuracy on.

                Returns:
                    float: The accuracy of the model on the dataset.

                Note:
                    The accuracy is calculated by comparing the model's predictions with the true labels in the dataset.
                """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    data = read_csv('../../../datasets/iris/iris.csv', sep=',', features=True, label=True)
    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = RandomForestClassifier(n_estimators=1000, max_features=4, min_sample_split=2, max_depth=5, mode='gini',
                                   seed=42)
    model.fit(train)
    print(model.score(test))