# regpy

Uma biblioteca leve, modular e eficiente desenvolvida em Python puro para **regressão estatística e aprendizado de máquina**.

A `regpy` está sendo construída sem dependências externas como NumPy ou SciPy, com o objetivo de implementar os algoritmos de regressão a partir de seus fundamentos matemáticos.

A biblioteca utiliza álgebra linear implementada internamente, incluindo resolução de sistemas de equações por **eliminação de Gauss-Jordan com pivoteamento parcial**.

> **Status:** projeto em desenvolvimento 🚧

---

## Sumário

* [Destaques](#destaques)
* [Instalação](#instalação)
* [Exemplo de Uso](#exemplo-de-uso)
* [Estrutura do Projeto](#estrutura-do-projeto)
* [Executando os Testes](#executando-os-testes)
* [Licença](#licença)

---

## Destaques

* **Zero dependências de terceiros:** o núcleo matemático é implementado utilizando apenas a biblioteca padrão do Python.
* **Álgebra linear própria:** operações necessárias para os modelos são desenvolvidas internamente.
* **Resolução de sistemas lineares:** implementação de eliminação de Gauss-Jordan com pivoteamento parcial.
* **Validação de dados:** verificação de tipos numéricos, valores infinitos, `NaN` e consistência das dimensões das matrizes.
* **Estrutura moderna em Python:** organização utilizando `src-layout`.
* **Testes automatizados:** utilização do `pytest` para verificar o comportamento dos modelos.
* **Arquitetura modular:** novos modelos de regressão podem ser adicionados sem comprometer os modelos existentes.

---

## Instalação

### Instalação diretamente pelo GitHub

A biblioteca pode ser instalada diretamente do repositório:

```bash
pip install git+https://github.com/erickmateusdev/regpy.git
```

### Instalação para desenvolvimento local

Clone o repositório:

```bash
git clone https://github.com/erickmateusdev/regpy.git
cd regpy
```

Instale o pacote em modo editável:

```bash
pip install -e .
```

Para instalar o `pytest`:

```bash
pip install pytest
```

O modo editável (`-e`) permite que alterações feitas no código-fonte sejam refletidas imediatamente no ambiente de desenvolvimento, sem precisar reinstalar o pacote.

---

## Exemplo de Uso

### Regressão Linear Múltipla

```python
from regpy.models.linear import Regressao_Linear

# Matriz de variáveis independentes (X)
X = [
    [1.0, 1.0],
    [2.0, 1.0],
    [1.0, 2.0],
    [3.0, 2.0],
    [4.0, 3.0]
]

# Vetor da variável dependente (y)
y = [10.0, 12.0, 13.0, 17.0, 22.0]

# Instanciação e treinamento do modelo
modelo = Regressao_Linear()

modelo.fit(X, y)

# Parâmetros aprendidos
print("Intercepto:", modelo.intercept)
print("Coeficientes:", modelo.coeficientes)

# Novos dados para predição
novos_dados = [
    [6.0, 1.0],
    [10.0, 3.0]
]

previsoes = modelo.predict(novos_dados)

print("Predições:", previsoes)
```

O modelo estima uma equação do tipo:

$$
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n
$$

onde:

* `β₀` é o intercepto;
* `β₁, β₂, ..., βₙ` são os coeficientes das variáveis independentes;
* `x₁, x₂, ..., xₙ` são as variáveis utilizadas como entrada.

---

## Estrutura do Projeto

```text
regpy/
│
├── pyproject.toml
├── README.md
├── LICENSE
│
├── src/
│   └── regpy/
│       ├── __init__.py
│       │
│       └── models/
│           ├── __init__.py
│           └── linear.py
│
└── tests/
    └── test_linear.py
```

A estrutura utiliza o padrão **`src-layout`**, separando o código da biblioteca dos testes e dos arquivos de configuração do projeto.

---

## Executando os Testes

A biblioteca possui testes automatizados para verificar tanto a precisão matemática dos modelos quanto o comportamento diante de entradas inválidas.

Para executar todos os testes:

```bash
python -m pytest
```

Para obter uma saída mais detalhada:

```bash
python -m pytest -v
```

Exemplo de resultado:

```text
collected 1 item

tests/test_linear.py .    [100%]

1 passed
```

---

## Licença

Este projeto está disponível sob a **Licença MIT**.

Você pode estudar, modificar e reutilizar o código de acordo com os termos da licença.
