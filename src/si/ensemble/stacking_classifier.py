import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier:
    """
    Ensemble classifier that combines the predictions of multiple base classifiers to make a final prediction, by
    training a second-level "meta-classifier" to make the final prediction using the output of the base classifiers as
    input.

    Parameters
    ----------
    models : array-like, shape = [n_models]
        Different models for the ensemble.
    final_model : The final model to make the final prediction.
    """
    def __init__(self, models: list, final_model: np.array):
        """
        Initialize the ensemble stacking classifier.

        :param models: array-like, shape = [n_models]
            Different models for the ensemble.
        :param final_model: np.array
            The final model to make the final prediction.
        """
        # parameters
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models to the dataset.

        :param dataset: Dataset object to fit the models to.

        :return: self: StackingClassifier
        """
        # training the models
        for model in self.models:
            model.fit(dataset)

        # getting the models' predictions
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # training the final model
        self.final_model.fit(Dataset(dataset.X, np.array(predictions).T))

        return self

    def _predict(self, dataset: Dataset) -> np.array:
        """
        Computes the prevision of all the models and returns the final model prediction.

        :param dataset: Dataset object to predict the labels of.

        :return: the final model prediction
        """
        # gets the model predictions
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # gets the final model previsions
        y_pred = self.final_model.predict(Dataset(dataset.X, np.array(predictions).T))

        return y_pred

    def _score(self, dataset: Dataset) -> float:
        """
        Calculates the accuracy of the model.
        :return: Accuracy of the model.
        """
        y_pred = self.predict(dataset)
        score = accuracy(dataset.y, y_pred)

        return score


if __name__ == '__main__':
    # imports
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.statistics.euclidean_distance import euclidean_distance

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN, Logistic classifier and final model
    knn = KNNClassifier(k=3)
    lg_model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    final_model = KNNClassifier(k=2, distance=euclidean_distance)

    # initialize the stacking classifier
    stacking = StackingClassifier([knn, lg_model], final_model)

    stacking.fit(dataset_train)

    # compute the score
    score = stacking.score(dataset_test)
    print(f"Score: {score}")

    # predictions
    print(stacking.predict(dataset_test))