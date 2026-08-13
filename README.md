# Hard Cardinality Constraints: Computational Experiments

This repository contains Python code for my MSc dissertation project on optimization problems with hard cardinality constraints.

The main aim is to compare different solution methods on two general cases:

1. Sparse portfolio optimization
2. Sparse linear regression / feature selection

The project focuses on how different methods deal with the requirement that only a limited number of variables can be selected.

---

## 1. Project Direction

The dissertation is method-focused rather than only application-focused. The main topic is hard cardinality constraints.

In sparse portfolio optimization, the cardinality constraint limits the number of selected assets in a portfolio.

In sparse linear regression, the cardinality constraint limits the number of selected features in the regression model.

The computational comparison will consider solution quality, sparsity, runtime, and optimality gap when a solver-based method is used.

---

## 2. General Cases and Methods

### 2.1 Sparse Portfolio Optimization

The portfolio problem is based on the mean-variance portfolio model. The sparse version adds a cardinality constraint so that at most (K) assets can be selected.

Planned methods:

* No-sparsity mean-variance portfolio model
* Gurobi MIP / MIQP cardinality-constrained portfolio model
* Genetic Algorithm heuristic
* Simulated Annealing heuristic

The Gurobi model is used as the main solver-based method because it can model the hard cardinality constraint directly with binary variables.

The heuristic methods are used as comparison methods because they can search for good sparse portfolios without solving the full MIP / MIQP model exactly.

### 2.2 Sparse Linear Regression / Feature Selection

The sparse regression problem aims to fit a regression model using only a limited number of features.

Planned methods:

* Ordinary least squares without sparsity
* Gurobi MIQP best subset selection
* LASSO
* Iterative Hard Thresholding

The Gurobi MIQP model is used as the main solver-based method because it directly controls the number of selected features.

LASSO is used as an (L_1) relaxation baseline.

IHT is used as a hard-thresholding method because it directly keeps a (k)-sparse solution during the algorithm.

---

## 3. Datasets

### 3.1 Portfolio Optimization Datasets

Planned datasets:

* OR-Library port1--port5
* Yahoo Finance / S&P 500 sample

The OR-Library datasets are used as benchmark portfolio datasets. They include expected returns, standard deviations, and correlations between assets.

Yahoo Finance data is used as a more realistic market data source. Historical stock prices need to be downloaded first. Then returns, mean returns, and covariance matrices are calculated.

### 3.2 Sparse Regression / Feature Selection Datasets

Planned datasets:

* Diabetes dataset
* Gisette
* Communities and Crime
* 197_cpu_act from PMLB

The Diabetes dataset is used as a small starting dataset.

Gisette is used as a high-dimensional feature selection dataset.

Communities and Crime is used as a real regression dataset with more predictors.

197_cpu_act is used as a medium-size regression dataset from PMLB.

---

## 4. Project Structure

```text
cardinality_dissertation_code/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── portfolio_data.py
│   ├── portfolio_gurobi.py
│   ├── portfolio_heuristics.py
│   ├── regression_data.py
│   ├── regression_gurobi.py
│   ├── regression_lasso.py
│   ├── regression_iht.py
│   └── metrics.py
│
├── experiments/
│   ├── run_portfolio_yahoo_test.py
│   ├── run_portfolio_orlibrary.py
│   ├── run_regression_diabetes.py
│   └── run_regression_experiments.py
│
├── results/
│   ├── tables/
│   └── figures/
│
├── README.md
└── requirements.txt
```

---

## 5. Installation

Install the required Python packages:

```bash
pip install numpy pandas scikit-learn matplotlib yfinance pmlb gurobipy
```

Gurobi also requires a valid academic license.

To check whether Gurobi is working:

```python
import gurobipy as gp

model = gp.Model()
print("Gurobi is working")
```

---

## 6. Planned Experiments

### Portfolio Experiments

The portfolio experiments will first be tested on a small Yahoo Finance sample.

The planned workflow is:

1. Download stock prices
2. Calculate returns
3. Estimate mean return vector and covariance matrix
4. Solve the portfolio problem without sparsity
5. Solve the sparse portfolio problem using Gurobi
6. Solve the sparse portfolio problem using Genetic Algorithm
7. Solve the sparse portfolio problem using Simulated Annealing
8. Compare the results

Comparison metrics:

* Portfolio variance
* Portfolio risk
* Expected return
* Number of selected assets
* Runtime
* Gurobi optimality gap

### Regression Experiments

The regression experiments will first be tested on the Diabetes dataset.

The planned workflow is:

1. Load dataset
2. Standardize features
3. Split into training and testing data
4. Solve ordinary least squares without sparsity
5. Solve sparse regression using Gurobi MIQP
6. Solve LASSO
7. Solve IHT
8. Compare the results

Comparison metrics:

* Training MSE
* Test MSE
* Number of selected features
* Runtime
* Gurobi optimality gap

---

## 7. Literature Connection

The portfolio part is mainly related to cardinality-constrained portfolio optimization and sparse portfolio selection.

The regression part is mainly related to best subset selection, LASSO-type methods, and hard-thresholding methods.

The selected papers are used as representative examples of different method families rather than a complete review of all papers in each area.

Main papers include:

* Cardinality Minimization, Constraints, and Regularization: A Survey
* Heuristics for Cardinality Constrained Portfolio Optimisation
* A Scalable Algorithm for Sparse Portfolio Selection
* Best Subset Selection via a Modern Optimization Lens
* Sensitivity of (L_1) Minimization to Parameter Choice
* The Trimmed Lasso
* Fast Iterative Hard Thresholding Methods with Pruning Gradient Computations
