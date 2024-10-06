from scipy import stats
import scipy
from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> tuple:
    """
    Scoring function for classification problems. It computes one-way ANOVA F-value for the
    provided dataset. The F-value scores allows analyzing if the mean between two or more groups (factors)
    are significantly different. Samples are grouped by the labels of the dataset.

    Parameters
    ----------
    dataset: Dataset
        A labeled dataset

    Returns
    -------
    F: np.array, shape (n_features,)
        F scores
    p: np.array, shape (n_features,)
        p-values
    """
    classes = dataset.get_classes()

    groups = []
    for class_ in classes:
        mask = dataset.y == class_
        group = dataset.X[mask,:]
        groups.append(group)
    return scipy.stats.f_oneway(*groups)
    