from regpy.models.linear import Linear_regression
import pytest

def test_predict_with_infinity():
    X = [
        [1],
        [2],
        [3],
        [4]
    ]

    y = [5, 7, 9, 11]

    modelo = Linear_regression()
    modelo.fit(X, y)

    valores = [
        float("inf"),
        float("-inf")
    ]

    for valor in valores:
        with pytest.raises(ValueError):
            modelo.predict([[valor]])

def test_predict_with_nan():
    X = [
        [1],
        [2],
        [3],
        [4]
    ]

    y = [5, 7, 9, 11]

    modelo = Linear_regression()
    modelo.fit(X, y)

    with pytest.raises(ValueError):
        modelo.predict([[float("nan")]])

def test_residual_orthogonality_multiple_regression():
    X = [
        [1, 1],
        [2, 1],
        [1, 2],
        [3, 2],
        [4, 3]
    ]

    y = [10, 12, 13, 17, 22]

    modelo = Linear_regression()
    modelo.fit(X, y)

    previsoes = modelo.predict(X)

    residuos = [
        observado - previsto
        for observado, previsto in zip(y, previsoes)
    ]

    for j in range(len(X[0])):
        soma_produtos = sum(
            X[i][j] * residuos[i]
            for i in range(len(X))
        )

        assert abs(soma_produtos) < 1e-10

def test_residual_orthogonality():
    X = [
        [1],
        [2],
        [3],
        [4],
        [5]
    ]

    y = [8, 11, 15, 16, 21]

    modelo = Linear_regression()
    modelo.fit(X, y)

    previsoes = modelo.predict(X)

    residuos = [
        observado - previsto
        for observado, previsto in zip(y, previsoes)
    ]

    soma_produtos = sum(
        X[i][0] * residuos[i]
        for i in range(len(X))
    )

    assert abs(soma_produtos) < 1e-10

def test_residual_sum():
    X = [
        [1],
        [2],
        [3],
        [4],
        [5]
    ]

    y = [8, 11, 15, 16, 21]

    modelo = Linear_regression()
    modelo.fit(X, y)

    previsoes = modelo.predict(X)

    residuos = [
        observado - previsto
        for observado, previsto in zip(y, previsoes)
    ]

    soma_residuos = sum(residuos)

    assert abs(soma_residuos) < 1e-10

def test_sample_without_features():
    X = [
        [1, 2],
        [],
        [3, 4]
    ]

    y = [10, 20, 30]

    modelo = Linear_regression()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_sample_not_iterable():
    X = [
        [1, 2],
        3,
        [3, 4]
    ]

    y = [10, 20, 30]

    modelo = Linear_regression()

    with pytest.raises(TypeError):
        modelo.fit(X, y)

def test_number_overflow():
    X = [
        [1],
        [10**10000],
        [3]
    ]

    y = [10, 20, 30]

    modelo = Linear_regression()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_y_with_non_finite_values():
    valores = [
        float("nan"),
        float("inf"),
        float("-inf")
    ]

    for valor in valores:
        X = [
            [1],
            [2],
            [3]
        ]

        y = [10, valor, 30]

        modelo = Linear_regression()

        with pytest.raises(ValueError):
            modelo.fit(X, y)

def test_y_with_non_numeric_values():
    X = [
        [1],
        [2],
        [3]
    ]

    y = [10, "abc", 30]

    modelo = Linear_regression()

    with pytest.raises(TypeError):
        modelo.fit(X, y)

def test_multiple_predictions():
    X = [
        [1],
        [2],
        [3],
        [4]
    ]

    y = [5, 7, 9, 11]

    modelo = Linear_regression()
    modelo.fit(X, y)

    previsoes = modelo.predict([
        [10],
        [5],
        [0],
        [-2]
    ])

    esperados = [23, 13, 3, -1]

    assert len(previsoes) == len(esperados)

    for previsao, esperado in zip(previsoes, esperados):
        assert abs(previsao - esperado) < 1e-10

def test_regression_with_extreme_values():
    X = [
        [100000],
        [200000],
        [300000],
        [400000]
    ]

    y = [200010, 400010, 600010, 800010]

    modelo = Linear_regression()
    modelo.fit(X, y)

    previsoes = modelo.predict([[500000]])

    assert abs(modelo.intercept - 10) < 1e-6
    assert abs(modelo.coeficientes[0] - 2) < 1e-10
    assert abs(previsoes[0] - 1000010) < 1e-6

def test_regression_with_two_observations():
    X = [
        [1],
        [2]
    ]

    y = [7, 9]

    modelo = Linear_regression()
    modelo.fit(X, y)

    previsoes = modelo.predict([[3]])

    assert abs(modelo.intercept - 5) < 1e-10
    assert abs(modelo.coeficientes[0] - 2) < 1e-10
    assert abs(previsoes[0] - 11) < 1e-10

def test_regression_with_negative_values():
    X = [
        [-3],
        [-2],
        [-1],
        [0],
        [1]
    ]

    y = [11, 8, 5, 2, -1]

    modelo = Linear_regression()
    modelo.fit(X, y)

    previsoes = modelo.predict([[-4], [2]])

    assert abs(modelo.intercept - 2) < 1e-10
    assert abs(modelo.coeficientes[0] - (-3)) < 1e-10

    assert abs(previsoes[0] - 14) < 1e-10
    assert abs(previsoes[1] - (-4)) < 1e-10

def test_simple_linear_regression():
    X = [
        [1],
        [2],
        [3],
        [4],
        [5]
    ]

    y = [7, 10, 13, 16, 19]

    modelo = Linear_regression()
    modelo.fit(X, y)

    previsoes = modelo.predict([[6], [7]])

    assert abs(modelo.intercept - 4) < 1e-10
    assert abs(modelo.coeficientes[0] - 3) < 1e-10

    assert abs(previsoes[0] - 22) < 1e-10
    assert abs(previsoes[1] - 25) < 1e-10

def test_constant_feature():
    X = [
        [5, 1],
        [5, 2],
        [5, 3],
        [5, 4]
    ]

    y = [10, 15, 20, 25]

    modelo = Linear_regression()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_singular_matrix():
    X = [
        [1, 2],
        [2, 4],
        [3, 6],
        [4, 8]
    ]

    y = [10, 20, 30, 40]

    modelo = Linear_regression()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_predict_incompatible_dimensions():
    X = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ]

    y = [10, 15, 20, 25]

    modelo = Linear_regression()
    modelo.fit(X, y)

    with pytest.raises(ValueError):
        modelo.predict([[1, 2, 3]])

def test_empty_y_vector():
    X = [
        [1, 2],
        [2, 3],
        [3, 4]
    ]

    y = []

    modelo = Linear_regression()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_empty_matrix():
    X = []
    y = []

    modelo = Linear_regression()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_infinite_values():
    valores = [
        float("inf"),
        float("-inf")
    ]

    for valor in valores:
        X = [
            [1, 2],
            [2, valor],
            [3, 4]
        ]

        y = [10, 20, 30]

        modelo = Linear_regression()

        with pytest.raises(ValueError):
            modelo.fit(X, y)


def test_boolean_values():
    X = [
        [1, 2],
        [2, True],
        [3, 4]
    ]

    y = [10, 20, 30]

    modelo = Linear_regression()

    with pytest.raises(TypeError):
        modelo.fit(X, y)

def test_nan_values():
    X = [
        [1, 2],
        [2, float("nan")],
        [3, 4]
    ]

    y = [10, 20, 30]

    modelo = Linear_regression()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_non_numeric_values():
    X = [
        [1, 2],
        [2, "abc"],
        [3, 4]
    ]

    y = [10, 20, 30]

    modelo = Linear_regression()

    with pytest.raises(TypeError):
        modelo.fit(X, y)

def test_irregular_matrix():
    X = [
        [1, 2],
        [2, 3],
        [3, 4, 5]
    ]

    y = [10, 20, 30]

    modelo = Linear_regression()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_incompatible_dimensions():
    X = [
        [1, 2],
        [2, 3],
        [3, 4]
    ]

    y = [10, 20]

    modelo = Linear_regression()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_predict_before_training():
    modelo = Linear_regression()

    with pytest.raises(ValueError):
        modelo.predict([[1, 2]])

def test_multiple_linear_regression():
    X = [
        [1, 1],
        [2, 1],
        [1, 2],
        [3, 2],
        [4, 3]
    ]

    y = [10, 12, 13, 17, 22]

    modelo = Linear_regression()
    modelo.fit(X, y)

    previsoes = modelo.predict([[6, 1], [10, 3]])

    assert abs(modelo.intercept - 5) < 1e-10
    assert abs(modelo.coeficientes[0] - 2) < 1e-10
    assert abs(modelo.coeficientes[1] - 3) < 1e-10

    assert abs(previsoes[0] - 20) < 1e-10
    assert abs(previsoes[1] - 34) < 1e-10

def test_refit_updates_model_parameters():
    modelo = Linear_regression()
    modelo.fit([[1], [2], [3]], [3, 5, 7])

    modelo.fit([[1], [2], [3]], [10, 7, 4])

    assert modelo.intercept == pytest.approx(13)
    assert modelo.coeficientes == pytest.approx([-3])
    assert modelo.predict([[4]]) == pytest.approx([1])

def test_failed_refit_invalidates_previous_model():
    modelo = Linear_regression()
    modelo.fit([[1], [2], [3]], [3, 5, 7])

    with pytest.raises(ValueError):
        modelo.fit([[1, 2], [2, 4], [3, 6]], [3, 6, 9])

    with pytest.raises(ValueError, match="needs to be fitted"):
        modelo.predict([[4]])

def test_predict_with_empty_matrix():
    modelo = Linear_regression()
    modelo.fit([[1], [2], [3]], [3, 5, 7])

    with pytest.raises(ValueError, match="at least one sample"):
        modelo.predict([])

def test_predict_with_irregular_matrix():
    modelo = Linear_regression()
    modelo.fit([[1], [2], [3]], [3, 5, 7])

    with pytest.raises(ValueError, match="must have 1 features"):
        modelo.predict([[1], [2, 3]])

@pytest.mark.parametrize(
    ("X", "y"),
    [
        ([[1e308], [5e307], [2.5e307]], [3, 2, 1]),
        ([[1], [2], [3]], [1e308, 0, -1e308]),
    ],
)
def test_fit_rejects_non_finite_intermediate_values(X, y):
    with pytest.raises(ValueError, match="finite"):
        Linear_regression().fit(X, y)