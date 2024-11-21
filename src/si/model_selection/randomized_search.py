from typing import Dict, Tuple, Callable, Union
import numpy as np
from si.data.dataset import Dataset
from si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(model,
                         dataset: Dataset,
                         parameter_distribution: Dict[str, Tuple],
                         scoring: Callable = None,
                         cv: int = 3,
                         n_iter: int = 10,
                         test_size: float = 0.3) -> Dict[str, Tuple[str, Union[int, float]]]:
    """
    Perform a randomized search over hyperparameters and evaluate model performance.

    param model: Model to validate
    param dataset: Validation dataset
    param parameter_distribution: Dictionary with hyperparameter names and their possible values
    param scoring: Scoring function
    param cv: Number of folds
    param n_iter: Number of random hyperparameter combinations to test
    param test_size: Test set size

    return: Dictionary with the results of the randomized search cross-validation.
    """
    scores = {'parameters': [], 'seed': [], 'train': [], 'test': []}

    # checks if parameters exist in the model
    for parameter in parameter_distribution:
        if not hasattr(model, parameter):
            raise AttributeError(f"The {model} does not have parameter {parameter}.")

    # sets n_iter parameter combinations
    for i in range(n_iter):

        # set the random seed
        random_state = np.random.randint(0, 1000)

        # store the seed
        scores['seed'].append(random_state)

        # dictionary for the parameter configuration
        parameters = {}

        # select the parameters and its value
        for parameter, value in parameter_distribution.items():
            parameters[parameter] = np.random.choice(value)

        # set the parameters to the model
        for parameter, value in parameters.items():
            setattr(model, parameter, value)

        # get scores from cross_validation
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # stores the parameter combination and the obtained score to the dictionary
        scores['parameters'].append(parameters)
        scores['train'].append(score)
        scores['test'].append(score)

    return scores
