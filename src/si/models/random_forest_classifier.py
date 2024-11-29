import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.metrics.accuracy import accuracy


class RandomForestClassifier(Model):
    """
    RandomForestClassifier is  model based on ensemble that combines multiple decision trees
    to improve the accuracy and robustness and reduce overfitting.
    """
    def __init__(self, n_estimators=100, max_features=None, min_sample_split=2, max_depth=10, 
                 mode="gini", seed=None, **kwargs):
        """
        Parameters:
        -----------
        n_estimators: int
            Number of decision trees in the ensemble.
        max_features: int
            Maximum number of features to consider in each tree.
        min_sample_split: int
            Minimum number of samples allowed in a decision node.
        max_depth: int
            Maximum depth of the decision trees.
        mode: str
            Impurity calculation mode (“gini” or “entropy”).
        seed: int
            Seed for randomization control.

        """
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = [] 

    def _fit(self, dataset: Dataset):
        """
        Trains the RandomForestClassifier using bootstrap samples from the dataset.

        Parameters:
        -----------
        dataset: Dataset
            Dataset for training.

        Return:
        --------
        self: RandomForestClassifier
            The trained model.
        """
        # Configurar semente aleatória
        if self.seed is not None:
            np.random.seed(self.seed)

        # Definir max_features caso seja None
        n_samples, n_features = dataset.X.shape
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            # Criar bootstrap dataset
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)

            bootstrap_X = dataset.X[bootstrap_indices][:, feature_indices]
            bootstrap_y = dataset.y[bootstrap_indices]

            # Treinar árvore de decisão
            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split, max_depth=self.max_depth,mode=self.mode)
            tree.fit(Dataset(X=bootstrap_X, y=bootstrap_y))

            # Armazenar a árvore treinada e as features usadas
            self.trees.append((feature_indices, tree))

        return self

    def _predict(self, dataset: Dataset)->np.ndarray:
        """
        Predicts the dataset labels using majority voting of the trees.

        Parameters:
        -----------
        dataset: Dataset
            Dataset for prediction.

        Return:
        --------
        predictions: np.ndarray
            Labels predicted by the model.
        """
        tree_predictions = []

        # Obter as predições de cada árvore
        for feature_indices, tree in self.trees:
            selected_features = dataset.X[:, feature_indices]
            sub_dataset = Dataset(X=selected_features, y=None)  
            predictions = tree.predict(sub_dataset)
            tree_predictions.append(predictions)

        tree_predictions = np.array(tree_predictions).T  # (amostras, árvores)
        final_predictions = []
        for row in tree_predictions:
            # Obter rótulos únicos e suas contagens
            unique, counts = np.unique(row, return_counts=True)
            # Selecionar o rótulo com maior contagem
            majority_vote = unique[np.argmax(counts)]
            final_predictions.append(majority_vote)

        return np.array(final_predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Evaluates the accuracy of the model on the dataset.

        Parameters:
        -----------
        dataset: Dataset
            Dataset for evaluation.
        predictions: np.ndarray
            Predictions

        Return:
        --------
        accuracy_score: float
            Model accuracy.
        """
        return accuracy(dataset.y, predictions)
