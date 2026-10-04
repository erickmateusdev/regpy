from regpy.models.linear import Regressao_Linear


def test_regressao_linear_multipla():
    X = [
        [1, 1],
        [2, 1],
        [1, 2],
        [3, 2],
        [4, 3]
    ]

    y = [10, 12, 13, 17, 22]

    modelo = Regressao_Linear()
    modelo.fit(X, y)

    previsoes = modelo.predict([[6, 1], [10, 3]])

    assert abs(modelo.intercept - 5) < 1e-10
    assert abs(modelo.coeficientes[0] - 2) < 1e-10
    assert abs(modelo.coeficientes[1] - 3) < 1e-10

    assert abs(previsoes[0] - 20) < 1e-10
    assert abs(previsoes[1] - 34) < 1e-10