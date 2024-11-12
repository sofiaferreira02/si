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
    Método para otimizar parâmetros usando combinações aleatórias.
    Avalia apenas um conjunto aleatório de parâmetros retirados de uma distribuição ou conjunto de valores possíveis.
    É mais eficiente do que o grid search e consegue encontrar melhores valores de hiperparâmetros.

    :param model: Modelo a validar
    :param dataset: Dataset de validação
    :param parameter_distribution: Os parrâmetros para a procura. Dicionário com nome do parâmetro e distribuição de
    valores
    :param scoring: Função de score
    :param cv: Número de folds
    :param n_iter: Número de combinações aleatórias de parâmetros
    :param test_size: Tamanho do dataset de teste

    :return: Lista de dicionários com a combinação dos parâmetros e os scores de treino e teste
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


if __name__ == '__main__':
    # imports
    from si.io.csv_file import read_csv
    from sklearn.preprocessing import StandardScaler
    from si.models.logistic_regression import LogisticRegression

    # read and standardize the dataset
    breast_bin = read_csv('../../../datasets/breast_bin/breast-bin.csv', sep=',', features=True, label=True)
    dataset = read_csv(breast_bin, sep=",", label=True)
    dataset.X = StandardScaler().fit_transform(dataset.X)

    # initialize the randomized search
    lg_model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    lg_model_param = {'l2_penalty': np.linspace(1, 10, 10),
                      'alpha': np.linspace(0.001, 0.0001, 100),
                      'max_iter': np.linspace(1000, 2000, 200)}
    scores = randomized_search_cv(lg_model, dataset, lg_model_param, cv=3)
    print(f'Scores: ', scores)