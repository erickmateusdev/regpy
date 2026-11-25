# regpy

A lightweight, modular, and efficient library developed in pure Python for **statistical regression and machine learning**.

`regpy` is being built without external dependencies such as NumPy or SciPy, with the goal of implementing regression algorithms from their mathematical foundations.

The library uses internally implemented linear algebra, including solving systems of equations through **Gauss-Jordan elimination with partial pivoting**.

> **Status:** project under development 🚧

---

## Table of Contents

* [Highlights](#highlights)
* [Installation](#installation)
* [Usage Example](#usage-example)
* [Project Structure](#project-structure)
* [Running the Tests](#running-the-tests)
* [License](#license)

---

## Highlights

* **Zero third-party dependencies:** the mathematical core uses only the Python standard library.
* **In-house linear algebra:** operations required by the models are implemented internally.
* **Linear system solver:** implementation of Gauss-Jordan elimination with partial pivoting.
* **Data validation:** checks numeric types, infinite values, `NaN`, and matrix dimension consistency.
* **Modern Python structure:** organized using the `src-layout`.
* **Automated tests:** `pytest` is used to verify model behavior.
* **Modular architecture:** new regression models can be added without affecting existing models.

---

## Installation

### Install directly from GitHub

The library can be installed directly from the repository:

```bash
pip install git+https://github.com/erickmateusdev/regpy.git
```

### Install for local development

Clone the repository:

```bash
git clone https://github.com/erickmateusdev/regpy.git
cd regpy
```

Install the package in editable mode:

```bash
pip install -e .
```

To install `pytest`:

```bash
pip install pytest
```

Editable mode (`-e`) allows changes to the source code to be reflected immediately in the development environment without reinstalling the package.

---

## Usage Example

### Multiple Linear Regression

```python
from regpy.models.linear import Linear_regression

# Independent variable matrix (X)
X = [
    [1.0, 1.0],
    [2.0, 1.0],
    [1.0, 2.0],
    [3.0, 2.0],
    [4.0, 3.0]
]

# Dependent variable vector (y)
y = [10.0, 12.0, 13.0, 17.0, 22.0]

# Create and train the model
model = Linear_regression()
model.fit(X, y)

# Learned parameters
print("Intercept:", model.intercept)
print("Coefficients:", model.coeficientes)

# New data for prediction
new_data = [
    [6.0, 1.0],
    [10.0, 3.0]
]

predictions = model.predict(new_data)

print("Predictions:", predictions)
```

The model estimates an equation of the form:

$$
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_nx_n
$$

where:

* `β₀` is the intercept;
* `β₁, β₂, ..., βₙ` are the coefficients of the independent variables;
* `x₁, x₂, ..., xₙ` are the input variables.

---

## Project Structure

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

The project uses the **`src-layout`** pattern, separating the library code from tests and project configuration files.

---

## Running the Tests

The library includes automated tests to verify both the mathematical accuracy of the models and their behavior with invalid input.

To run all tests:

```bash
python -m pytest
```

For more detailed output:

```bash
python -m pytest -v
```

Example output:

```text
collected 34 items

tests/test_linear.py .................................. [100%]

34 passed
```

---

## License

This project is available under the **MIT License**.

You may study, modify, and reuse the code according to the terms of the license.
