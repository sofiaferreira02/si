from abc import ABCMeta, ABC, abstractmethod

from si.base.estimator import Estimator
from si.data.dataset import Dataset
from numpy import np


class Model(Estimator, ABC):
    """
    Abstract base class for models.
    A model is an object that can predict the target values of a Dataset object.
    """

    def __init__(self, **kwargs):
        """
        Initialize the model.
        """
        super().__init__(**kwargs)

    def predict(self, dataset):
        """
        Predict the target values of the dataset.
        The model needs to be fitted before calling this method.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """
        if not self.is_fitted:
            raise ValueError('Model needs to be fitted before calling predict()')
        return self._predict(dataset)

    @abstractmethod
    def _predict(self, dataset):
        """
        Predict the target values of the dataset.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """

    def fit_predict(self, dataset):
        """
        Fit the model to the dataset and predict the target values.
        Equivalent to calling fit(dataset) and then predict(dataset).

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit and predict the target values of.

        Returns
        -------
        predictions: np.ndarray
            The predicted target values.
        """
        self.fit(dataset)
        return self.predict(dataset)
    
    @abstractmethod
    def _score(self, dataset:Dataset, predictions: np.ndarray) -> float:
        """
        Compute the score of the model on the dataset.
        Abstract method that needs to be implemented by all subclasses.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score on.

        Returns
        -------
        score: float
            The computed score of the model.
        """

    def score(self, dataset:Dataset) -> float:
        """
        Compute the score of the model on the dataset.
        The model needs to be fitted before calling this method.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score on.

        Returns
        -------
        score: float
            The computed score of the model.
        """
        if self.is_fitted():

            predictions =self.predict(dataset=dataset)
            self._score(dataset, predictions=predictions)
        else: 
            raise ValueError('Your model is not fitted, please call method.fit()')
