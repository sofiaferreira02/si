import numpy as np
from si.metrics.mse import mse
from si.data.dataset import Dataset  

class LassoRegression:
    """
    Lasso Regressor
    The Lasso regressor is a linear model that performs both feature selection and regularization 
    to enhance the prediction accuracy and interpretability of the model.
    
    Parameters
    ----------
    l1_penalty: float
        The L1 regularization parameter that controls the strength of the regularization.
    scale: bool
        Whether to scale the input data to have zero mean and unit variance. 

    Attributes
    ----------
    theta: np.ndarray
        The coefficients (weights) assigned to each feature in the model after fitting. These are learned during training.
    theta_zero: float
        The intercept term (also called the bias or zero coefficient), representing the value of the predicted variable
        when all features are zero.
    mean: np.ndarray
        The mean of each feature in the training data. Used for scaling if the `scale` parameter is set to True.
    std: np.ndarray
        The standard deviation of each feature in the training data. Used for scaling if the `scale` parameter is set to True.

        """


    def __init__(self, l1_penalty=1.0, scale=True, patience: int = 5):
        self.l1_penalty = l1_penalty
        self.scale = scale
        self.patience = patience

        self.theta = None
        self.theta_zero = 0
        self.mean = None
        self.std = None
        self.cost_history = []  # Armazena o histórico de custos

    def _fit(self, dataset: Dataset, max_iter=1000, tolerance=1e-4):
        X, y = dataset.X, dataset.y
        n, p = X.shape

        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std

        self.theta = np.zeros(p)
        self.theta_zero = 0
        self.cost_history = []

        for _ in range(max_iter):
            theta_old = self.theta.copy()

            for j in range(p):
                r_j = np.dot(X[:, j], (y - (self.theta_zero + np.dot(X, self.theta) - X[:, j] * self.theta[j])))
                self.theta[j] = self._soft_threshold(r_j, self.l1_penalty) / np.dot(X[:, j], X[:, j])

            self.theta_zero = np.mean(y - np.dot(X, self.theta))

            # Calcula e armazena o custo
            y_pred = np.dot(X, self.theta) + self.theta_zero
            cost = mse(y, y_pred)
            self.cost_history.append(cost)

            if np.max(np.abs(self.theta - theta_old)) < tolerance:
                break

    def _soft_threshold(self, r_j, l1_penalty):
        if r_j > l1_penalty:
            return r_j - l1_penalty
        elif r_j < -l1_penalty:
            return r_j + l1_penalty
        else:
            return 0

    def _predict(self, dataset: Dataset):
        X = dataset.X
        if self.scale:
            X = (X - self.mean) / self.std
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset):
        y_pred = self._predict(dataset)
        return mse(dataset.y, y_pred)