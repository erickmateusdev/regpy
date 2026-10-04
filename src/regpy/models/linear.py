import math
from numbers import Real

class Regressao_Linear:
    def __init__(self):
        self.coeficientes = []
        self.intercept = 0.0
        self.n_features = 0
        self._fitted = False

    def fit(self, X, y):
        self._fitted = False
        self.coeficientes = []
        self.intercept = 0.0
        self.n_features = 0

        X = self._validate_matrix(X, "X")
        if not y:
            raise ValueError("y precisa conter pelo menos um valor.")
        if len(X) != len(y):
            raise ValueError("X e y precisam conter o mesmo numero de amostras.")
        y = [self._validate_number(value, "y") for value in y]

        self.n_features = len(X[0])
        n_samples = len(X)

        # Adiciona uma coluna de 1s para calcular o intercepto junto aos coeficientes.
        design = [[1.0, *row] for row in X]
        n_parameters = self.n_features + 1
        matrix = [
            [
                sum(row[i] * row[j] for row in design)
                for j in range(n_parameters)
            ]
            for i in range(n_parameters)
        ]
        target = [
            sum(design[row][i] * y[row] for row in range(n_samples))
            for i in range(n_parameters)
        ]

        parameters = self._solve(matrix, target)
        self.intercept = parameters[0]
        self.coeficientes = parameters[1:]
        self._fitted = True
        return self

    @staticmethod
    def _validate_number(value, name):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Todos os valores de {name} precisam ser numericos.")
        try:
            number = float(value)
        except OverflowError as error:
            raise ValueError(f"Os valores de {name} precisam ser finitos.") from error
        if not math.isfinite(number):
            raise ValueError(f"Os valores de {name} precisam ser finitos.")
        return number

    @classmethod
    def _validate_matrix(cls, values, name, expected_features=None):
        if not values:
            raise ValueError(f"{name} precisa conter pelo menos uma amostra.")

        matrix = []
        for row in values:
            try:
                feature_count = len(row)
            except TypeError as error:
                raise TypeError(f"Cada amostra de {name} precisa ser uma lista de valores.") from error

            if feature_count == 0:
                raise ValueError(f"Cada amostra de {name} precisa conter pelo menos uma variavel.")
            if expected_features is not None and feature_count != expected_features:
                raise ValueError(
                    f"Cada amostra de {name} precisa ter {expected_features} variaveis."
                )

            matrix.append([cls._validate_number(value, name) for value in row])

        if expected_features is None and any(len(row) != len(matrix[0]) for row in matrix):
            raise ValueError(f"Todas as amostras de {name} precisam ter o mesmo numero de variaveis.")
        return matrix

    @staticmethod
    def _solve(matrix, target):
        size = len(target)
        augmented = [matrix[i][:] + [target[i]] for i in range(size)]

        for column in range(size):
            pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
            if augmented[pivot][column] == 0:
                raise ValueError(
                    "Nao foi possivel ajustar o modelo: variaveis redundantes "
                    "ou amostras insuficientes."
                )

            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            pivot_value = augmented[column][column]
            augmented[column] = [value / pivot_value for value in augmented[column]]

            for row in range(size):
                if row == column:
                    continue
                factor = augmented[row][column]
                augmented[row] = [
                    augmented[row][i] - factor * augmented[column][i]
                    for i in range(size + 1)
                ]

        return [augmented[i][-1] for i in range(size)]

    def predict(self, X):
        if not self._fitted:
            raise ValueError("O modelo precisa ser ajustado com fit() antes de predict().")
        X = self._validate_matrix(X, "X", self.n_features)

        return [
            self.intercept + sum(coef * value for coef, value in zip(self.coeficientes, row))
            for row in X
        ]