from typing import List
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier(Model):
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
    def __init__(self, base_models: List[Model], meta_model: Model, **kwargs):
        """
        Initialize the ensemble stacking classifier.

        Parameters
        ----------
        models : list
            Array-like of base models to be combined in the ensemble.
            Each model should be an instance of a Model class.
        final_model :
            Model to be used as the meta-model and create the final predictions.
            The model must be an instance of a Model class
        """
        # parameters
        super().__init__(**kwargs)
        self.base_models = base_models
        self.meta_model = meta_model
        self.meta_training_data = None 

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models to the dataset.

       Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model to (training dataset)

        Returns
        -------
        self : StackingClassifier
            The fitted model
        """
        for model in self.base_models:
            model.fit(dataset)

        # Gera as previsões dos modelos base
        base_predictions = np.array([model.predict(dataset) for model in self.base_models]).T

        # Cria o dataset de treinamento para o modelo final
        self.meta_training_data = Dataset(X=base_predictions, y=dataset.y, label=dataset.label)

        # Treina o modelo final
        self.meta_model.fit(self.meta_training_data)

        return self

    def _predict(self, dataset: Dataset) -> np.array:
        """
        Computes the prevision of all the models 

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict the labels for.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        # Gera previsões usando os modelos base
        base_predictions = np.array([model.predict(dataset) for model in self.base_models]).T

        # Cria um dataset para o modelo final com as previsões dos modelos base
        meta_input_data = Dataset(X=base_predictions, label=None)

        # Previsões finais com o modelo final
        return self.meta_model.predict(meta_input_data)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the accuracy of the model

        Parameters
        ----------
        dataset : Dataset
            The dataset to evaluate the model on.
        predictions : np.ndarray
            Predictions.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        return accuracy(dataset.y, predictions)

