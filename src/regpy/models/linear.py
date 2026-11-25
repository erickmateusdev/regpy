import math
from numbers import Real

class Linear_regression:
    def __init__(self):
        self.coefficients = []
        self.intercept = 0.0
        self.n_features = 0
        self._fitted = False

    def fit(self, X, y):
        self._fitted = False
        self.coefficients = []
        self.intercept = 0.0
        self.n_features = 0

        X = self._validate_matrix(X, "X")
        if not y:
            raise ValueError("Y must contain at least one value.")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
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

        if not all(math.isfinite(value) for row in matrix for value in row) or not all(
            math.isfinite(value) for value in target
        ):
            raise ValueError("Model calculations must produce finite values.")

        parameters = self._solve(matrix, target)
        if not all(math.isfinite(value) for value in parameters):
            raise ValueError("Model parameters must be finite.")
        self.intercept = parameters[0]
        self.coeficientes = parameters[1:]
        self._fitted = True
        return self

    @staticmethod
    def _validate_number(value, name):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"All values of {name} must be numeric.")
        try:
            number = float(value)
        except OverflowError as error:
            raise ValueError(f"Values of {name} must be finite.") from error
        if not math.isfinite(number):
            raise ValueError(f"Values of {name} must be finite.")
        return number

    @classmethod
    def _validate_matrix(cls, values, name, expected_features=None):
        if not values:
            raise ValueError(f"{name} must contain at least one sample.")

        matrix = []
        for row in values:
            try:
                feature_count = len(row)
            except TypeError as error:
                raise TypeError(f"Each sample of {name} must be a list of values.") from error

            if feature_count == 0:
                raise ValueError(f"Each sample of {name} must contain at least one feature.")
            if expected_features is not None and feature_count != expected_features:
                raise ValueError(
                    f"Each sample of {name} must have {expected_features} features."
                )

            matrix.append([cls._validate_number(value, name) for value in row])

        if expected_features is None and any(len(row) != len(matrix[0]) for row in matrix):
            raise ValueError(f"All samples of {name} must have the same number of features.")
        return matrix

    @staticmethod
    def _solve(matrix, target):
        size = len(target)
        augmented = [matrix[i][:] + [target[i]] for i in range(size)]

        for column in range(size):
            pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
            if augmented[pivot][column] == 0:
                raise ValueError(
                    "Could not fit the model: redundant variables or insufficient samples."
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
            raise ValueError("The model needs to be fitted with fit() before predict().")
        X = self._validate_matrix(X, "X", self.n_features)

        return [
            self.intercept + sum(coef * value for coef, value in zip(self.coeficientes, row))
            for row in X
        ]