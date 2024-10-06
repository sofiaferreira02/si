import numpy as np
from typing import Callable
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile:
    """
        Classe que integra os métodos responsáveis pela seleção de um percentil de features segundo a análise da
        variância(score_func), sendo que o percentil a selecionar é dado pelo utilizador.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values).
        percentile: int
            Percentile value.

        Attributes
        ----------
        F: array, shape (n_features,)
            F scores of features
        p: array, shape (n_features,)
            p-values of F-scores
        """

    def __init__(self, score_func: Callable = f_classification, percentile: int = 10):
        """
        Construtor

        :param score_func: função de análise da variância (f_classification() ou f_regression())
        :param percentile: valor do percentile. Apenas F-scores acima desse valor permanecem no dataset filtrado
        """
        if percentile > 100 or percentile < 0:
            raise ValueError("Your percentile must be between 0 and 100")

        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Method that selects the features with the highest F value in the desired percentile

        Parameters
        ----------
        dataset: Dataset, input Dataset


        Returns
        -------
        dataset: Dataset
            A labeled dataset with the select features.
        """
        len_features = len(dataset.features)
        features_percentile = int(
            len_features * (self.percentile / 100)  # calcula o nº de features selecionadas com F score
            # mais alto até ao valor do percentile indicado (50% de 10 features equivale a 5 features)
        )
        idxs = np.argsort(self.F)[-self.percentile:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        It fits SelectKBest to compute the F scores and p-values.

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

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Method that executes the fit method and then the transform method.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            Dataset with the selected variables
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    # dataset = read_csv('C:\Users\Bruna\PycharmProjects\pythonSIB2\SIB\src\si\IO\iris.csv', sep=",", label=True)
    select = SelectPercentile(percentile=50)
    # chamar o método f_classification p/ o cálculo e introduzir valor de
    # percentile (em percentagem)
    select = select.fit(dataset)
    dataset = select.transform(dataset)
    print(dataset.features)