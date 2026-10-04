-- regpy

Uma biblioteca leve, modular e eficiente desenvolvida em Python puro para regressão estatística e aprendizado de máquina, construída sem dependências externas (como NumPy ou SciPy).O objetivo da regpy é fornecer implementações limpas e didáticas de algoritmos de regressão, utilizando álgebra linear nativa com resolução de sistemas de equações via eliminação de Gauss-Jordan com pivoteamento parcial.

-- Sumário

- Destaques 
- Instalação
- Exemplo de Uso
- Estrutura do Projeto
- Executando os Testes
- Roadmap
- Licença

-- Destaques

- Zero Dependências de Terceiros: Todo o módulo de álgebra linear e manipulação de matrizes foi escrito do zero usando apenas bibliotecas padrão (math).
- Resolução Estável de Sistemas Lineares: Implementação de Eliminação de Gauss-Jordan com pivoteamento parcial para cálculo exato dos coeficientes e do intercepto.Validação Rigorosa de Dados: Checagem de tipos numéricos reais, valores infinitos/NaN e consistência de dimensões nas matrizes de entrada.
- Estrutura Moderna em Python: Organizado no formato src-layout com testes automatizados via pytest.

-- Instalação
- Instalação Direta via Pip (GitHub)Você pode instalar a biblioteca diretamente do GitHub em qualquer ambiente Python rodando:pip install git+https://github.com/erickmateusdev/regpy.git

-- Instalação para Desenvolvimento Local
- Se você deseja clonar e contribuir com o projeto:

# 1. Clone o repositório
git clone https://github.com/erickmateusdev/regpy.git
cd regpy

# 2. Instale o pacote em modo editável com dependências de desenvolvimento
pip install -e .
pip install pytest

-- Exemplo de Uso

- Regressão Linear

from regpy.models.linear import Regressao_Linear

# Matriz de variáveis independentes (X) e vetor alvo (y)
X = [
    [1.0, 1.0],
    [2.0, 1.0],
    [1.0, 2.0],
    [3.0, 2.0],
    [4.0, 3.0]
]
y = [10.0, 12.0, 13.0, 17.0, 22.0]

# Instanciação e treinamento do modelo
modelo = Regressao_Linear()
modelo.fit(X, y)

# Exibição dos parâmetros aprendidos
print("Intercepto:", modelo.intercept)
print("Coeficientes:", modelo.coeficientes)

# Realizando predições para novos dados
novos_dados = [
    [6.0, 1.0],
    [10.0, 3.0]
]
previsoes = modelo.predict(novos_dados)
print("Predições:", previsoes)


-- Estrutura do Projetoregpy/

├── src/
│   └── regpy/
│       ├── __init__.py
│       └── models/
│           ├── __init__.py
│           └── linear.py        # Algoritmo de Regressão Linear e Álgebra Linear
├── tests/
│   └── test_linear.py           # Testes unitários do modelo
├── pyproject.toml               # Configuração do pacote e metadados
├── .gitignore                   # Arquivos ignorados pelo Git
└── README.md                    # Documentação do projeto


-- Executando os Testes

A biblioteca conta com uma suíte de testes unitários para validar a precisão matemática e a robustez contra dados inválidos.
Para rodar os testes:

pytest

Para uma saída mais detalhada:

pytest -v

-- Licença: Este projeto está sob a licença MIT. Sinta-se livre para estudar, modificar e reutilizar o código.