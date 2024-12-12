import numpy as np
from typing import Callable
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Select features according to the highest scores up to a certain percentile.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    percentile: float, default=10
        Percentile of top features to select.

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float = 10, **kwargs):
        """
        Select features according to the highest scores up to a certain percentile

        Parameters
        ----------

        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: float, default=10
            Percentile of top features to select.
        
        """
        super().__init__(**kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset : Dataset) -> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self


    def _transform(self, dataset: Dataset)-> Dataset:
        """
        Select the top features based on the computed F-scores and the specified percentile.

        Parameters
        ----------
        dataset: Dataset
            The input dataset to transform.

        Returns
        -------
        transformed: Dataset
            A new dataset containing only the selected features.
        """
        # Compute the number of features to select
        num_features = int(len(dataset.features) * (self.percentile / 100))
        num_features = max(1, num_features)  # Ensure at least one feature is selected

        # Select indices of the top features
        top_indices = np.argsort(self.F)[-num_features:]

        # Subset the dataset
        selected_X = dataset.X[:, top_indices]
        selected_features = [dataset.features[i] for i in top_indices]

        return Dataset(X=selected_X, y=dataset.y, features=selected_features, label=dataset.label)



if __name__ == "__main__":
    from si.data.dataset import Dataset

    X = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
    y = np.array([0, 1, 0])
    features = ["a", "b", "c", "d"]
    label = "target"

    dataset = Dataset(X=X, y=y, features=features, label=label)
    selector = SelectPercentile(percentile=50)
    selector.fit(dataset)
    transformed_dataset = selector.transform(dataset)

    print("Features selecionadas:", transformed_dataset.features)