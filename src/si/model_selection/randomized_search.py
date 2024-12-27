from typing import Dict, Tuple, Callable
import numpy as np
from si.data.dataset import Dataset
from si.base.model import Model
from itertools import product
from si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(model: Model, 
                         dataset: Dataset,
                         hyperparameter_grid: Dict[str, Tuple],
                         scoring: Callable = None,
                         cv: int = 5,
                         n_iter: int = 10) -> Dict:
    """
    Perform a randomized search over hyperparameters and evaluate model performance.

    param model: Model to validate
    param dataset: Validation dataset
    param hyperparameter_grid: Dictionary with hyperparameter names and their possible values
    param scoring: Scoring function
    param cv: Number of folds
    param n_iter: Number of random hyperparameter combinations to test

    return: Dictionary with the results of the randomized search cross-validation.
    """
    results = {
        'hyperparameters': [],
        'scores': [],
        'best_hyperparameters': None,
        'best_score': -np.inf
    }

    # Validate hyperparameters
    for param in hyperparameter_grid:
        if not hasattr(model, param):
            raise AttributeError(f"Model {model} does not have a parameter '{param}'.")

    # Generate all possible combinations
    all_combinations = list(product(*hyperparameter_grid.values()))

    # Ensure n_iter does not exceed total combinations
    if n_iter > len(all_combinations):
        raise ValueError(f"n_iter cannot exceed the total number of combinations ({len(all_combinations)}).")

    # Randomly sample n_iter combinations
    sampled_combinations = [all_combinations[i] for i in np.random.choice(len(all_combinations), size=n_iter, replace=False)]

    # Initialize results dictionary
    results = {
        'hyperparameters': [],
        'scores': [],
        'best_hyperparameters': None,
        'best_score': -np.inf
    }

    # Iterate through sampled combinations
    for combination in sampled_combinations:
        # Map combination to hyperparameter names
        param_dict = {key: value for key, value in zip(hyperparameter_grid.keys(), combination)}

        # Set model parameters
        for param, value in param_dict.items():
            setattr(model, param, value)

        # Perform cross-validation
        cv_scores = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)
        mean_score = np.mean(cv_scores)

        # Update results
        results['hyperparameters'].append(param_dict)
        results['scores'].append(mean_score)

        # Update best score and parameters if applicable
        if mean_score > results['best_score']:
            results['best_score'] = mean_score
            results['best_hyperparameters'] = param_dict

    return results
