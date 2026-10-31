from regpy.models.linear import Regressao_Linear
import pytest

def test_predict_com_infinito():
    X = [
        [1],
        [2],
        [3],
        [4]
    ]

    y = [5, 7, 9, 11]

    modelo = Regressao_Linear()
    modelo.fit(X, y)

    valores = [
        float("inf"),
        float("-inf")
    ]

    for valor in valores:
        with pytest.raises(ValueError):
            modelo.predict([[valor]])

def test_predict_com_nan():
    X = [
        [1],
        [2],
        [3],
        [4]
    ]

    y = [5, 7, 9, 11]

    modelo = Regressao_Linear()
    modelo.fit(X, y)

    with pytest.raises(ValueError):
        modelo.predict([[float("nan")]])

def test_ortogonalidade_residuos_regressao_multipla():
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

def test_ortogonalidade_dos_residuos():
    X = [
        [1],
        [2],
        [3],
        [4],
        [5]
    ]

    y = [8, 11, 15, 16, 21]

    modelo = Regressao_Linear()
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

def test_soma_dos_residuos():
    X = [
        [1],
        [2],
        [3],
        [4],
        [5]
    ]

    y = [8, 11, 15, 16, 21]

    modelo = Regressao_Linear()
    modelo.fit(X, y)

    previsoes = modelo.predict(X)

    residuos = [
        observado - previsto
        for observado, previsto in zip(y, previsoes)
    ]

    soma_residuos = sum(residuos)

    assert abs(soma_residuos) < 1e-10

def test_amostra_sem_variaveis():
    X = [
        [1, 2],
        [],
        [3, 4]
    ]

    y = [10, 20, 30]

    modelo = Regressao_Linear()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_amostra_nao_iteravel():
    X = [
        [1, 2],
        3,
        [3, 4]
    ]

    y = [10, 20, 30]

    modelo = Regressao_Linear()

    with pytest.raises(TypeError):
        modelo.fit(X, y)

def test_numero_com_overflow():
    X = [
        [1],
        [10**10000],
        [3]
    ]

    y = [10, 20, 30]

    modelo = Regressao_Linear()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_y_com_valores_nao_finitos():
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

        modelo = Regressao_Linear()

        with pytest.raises(ValueError):
            modelo.fit(X, y)

def test_y_com_valores_nao_numericos():
    X = [
        [1],
        [2],
        [3]
    ]

    y = [10, "abc", 30]

    modelo = Regressao_Linear()

    with pytest.raises(TypeError):
        modelo.fit(X, y)

def test_multiplas_previsoes():
    X = [
        [1],
        [2],
        [3],
        [4]
    ]

    y = [5, 7, 9, 11]

    modelo = Regressao_Linear()
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

def test_regressao_com_valores_extremos():
    X = [
        [100000],
        [200000],
        [300000],
        [400000]
    ]

    y = [200010, 400010, 600010, 800010]

    modelo = Regressao_Linear()
    modelo.fit(X, y)

    previsoes = modelo.predict([[500000]])

    assert abs(modelo.intercept - 10) < 1e-6
    assert abs(modelo.coeficientes[0] - 2) < 1e-10
    assert abs(previsoes[0] - 1000010) < 1e-6

def test_regressao_com_duas_observacoes():
    X = [
        [1],
        [2]
    ]

    y = [7, 9]

    modelo = Regressao_Linear()
    modelo.fit(X, y)

    previsoes = modelo.predict([[3]])

    assert abs(modelo.intercept - 5) < 1e-10
    assert abs(modelo.coeficientes[0] - 2) < 1e-10
    assert abs(previsoes[0] - 11) < 1e-10

def test_regressao_com_valores_negativos():
    X = [
        [-3],
        [-2],
        [-1],
        [0],
        [1]
    ]

    y = [11, 8, 5, 2, -1]

    modelo = Regressao_Linear()
    modelo.fit(X, y)

    previsoes = modelo.predict([[-4], [2]])

    assert abs(modelo.intercept - 2) < 1e-10
    assert abs(modelo.coeficientes[0] - (-3)) < 1e-10

    assert abs(previsoes[0] - 14) < 1e-10
    assert abs(previsoes[1] - (-4)) < 1e-10

def test_regressao_linear_simples():
    X = [
        [1],
        [2],
        [3],
        [4],
        [5]
    ]

    y = [7, 10, 13, 16, 19]

    modelo = Regressao_Linear()
    modelo.fit(X, y)

    previsoes = modelo.predict([[6], [7]])

    assert abs(modelo.intercept - 4) < 1e-10
    assert abs(modelo.coeficientes[0] - 3) < 1e-10

    assert abs(previsoes[0] - 22) < 1e-10
    assert abs(previsoes[1] - 25) < 1e-10

def test_variavel_constante():
    X = [
        [5, 1],
        [5, 2],
        [5, 3],
        [5, 4]
    ]

    y = [10, 15, 20, 25]

    modelo = Regressao_Linear()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_matriz_singular():
    X = [
        [1, 2],
        [2, 4],
        [3, 6],
        [4, 8]
    ]

    y = [10, 20, 30, 40]

    modelo = Regressao_Linear()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_predict_dimensoes_incompativeis():
    X = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ]

    y = [10, 15, 20, 25]

    modelo = Regressao_Linear()
    modelo.fit(X, y)

    with pytest.raises(ValueError):
        modelo.predict([[1, 2, 3]])

def test_vetor_y_vazio():
    X = [
        [1, 2],
        [2, 3],
        [3, 4]
    ]

    y = []

    modelo = Regressao_Linear()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_matriz_vazia():
    X = []
    y = []

    modelo = Regressao_Linear()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_valores_infinitos():
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

        modelo = Regressao_Linear()

        with pytest.raises(ValueError):
            modelo.fit(X, y)


def test_valores_booleanos():
    X = [
        [1, 2],
        [2, True],
        [3, 4]
    ]

    y = [10, 20, 30]

    modelo = Regressao_Linear()

    with pytest.raises(TypeError):
        modelo.fit(X, y)

def test_valores_nan():
    X = [
        [1, 2],
        [2, float("nan")],
        [3, 4]
    ]

    y = [10, 20, 30]

    modelo = Regressao_Linear()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_valores_nao_numericos():
    X = [
        [1, 2],
        [2, "abc"],
        [3, 4]
    ]

    y = [10, 20, 30]

    modelo = Regressao_Linear()

    with pytest.raises(TypeError):
        modelo.fit(X, y)

def test_matriz_irregular():
    X = [
        [1, 2],
        [2, 3],
        [3, 4, 5]
    ]

    y = [10, 20, 30]

    modelo = Regressao_Linear()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_dimensoes_incompativeis():
    X = [
        [1, 2],
        [2, 3],
        [3, 4]
    ]

    y = [10, 20]

    modelo = Regressao_Linear()

    with pytest.raises(ValueError):
        modelo.fit(X, y)

def test_predict_sem_treinamento():
    modelo = Regressao_Linear()

    with pytest.raises(ValueError):
        modelo.predict([[1, 2]])

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