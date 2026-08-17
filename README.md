# Hard Cardinality Constraints: Computational Experiments

This repository contains the Python code for my MSc dissertation project on optimization problems with hard cardinality constraints.

The dissertation is method-focused. It studies how different exact, relaxation-based, and heuristic methods perform when only a limited number of variables are allowed to be selected.

Two computational cases are considered:

1. Sparse portfolio optimization
2. Sparse linear regression / feature selection

The main comparison focuses on solution quality, sparsity, runtime where available, and solver status or optimality gap for Gurobi-based models.

---

## 1. Project Direction

The main topic of this project is optimization with hard cardinality constraints.

A hard cardinality constraint directly limits the number of selected decision variables. This type of constraint appears in many sparse optimization problems, where the aim is not only to obtain a good objective value but also to produce a simpler and more interpretable solution.

In sparse portfolio optimization, the cardinality constraint limits the number of selected assets in a portfolio.

In sparse linear regression, the cardinality constraint limits the number of selected features in the regression model.

This repository implements and compares several methods for these two cases.

---

## 2. General Cases and Methods

### 2.1 Sparse Portfolio Optimization

The portfolio optimization problem is based on the mean-variance portfolio model. The sparse version adds a cardinality constraint so that at most K assets can be selected.

The implemented methods are:

* No-sparsity mean-variance portfolio model
* Gurobi cardinality-constrained portfolio model
* Genetic Algorithm heuristic
* Simulated Annealing heuristic

The Gurobi model is used as the main solver-based benchmark because it can model the hard cardinality constraint directly using binary variables.

The Genetic Algorithm and Simulated Annealing methods are used as heuristic comparison methods. They search over sparse asset subsets and then evaluate each selected subset through the corresponding restricted portfolio optimization problem.

### 2.2 Sparse Linear Regression / Feature Selection

The sparse regression problem aims to fit a regression model using only a limited number of features.

The implemented methods are:

* Ordinary least squares without sparsity
* LASSO
* Gurobi MIQP best subset selection
* Iterative Hard Thresholding

The Gurobi MIQP model is used as the direct hard-cardinality benchmark because it controls the number of selected features explicitly through binary variables.

LASSO is used as an L1 relaxation baseline.

Iterative Hard Thresholding is used as a hard-thresholding method because it directly keeps a K-sparse solution during the algorithm.

---

## 3. Datasets

### 3.1 Portfolio Optimization Datasets

The portfolio experiments use:

* OR-Library port1--port5 benchmark datasets
* Yahoo Finance stock price data

The OR-Library datasets include expected returns, standard deviations, and correlations between assets. These are used to construct the covariance matrix for the portfolio optimization models.

Yahoo Finance data is used as a market-data example. Historical adjusted closing prices are downloaded and converted into returns, mean returns, and covariance matrices.

### 3.2 Sparse Regression / Feature Selection Datasets

The sparse regression experiments use:

* Diabetes dataset
* 197_cpu_act dataset from PMLB
* Communities and Crime dataset

The Diabetes dataset is used as a small regression example.

The 197_cpu_act dataset is used as a medium-sized regression dataset.

The Communities and Crime dataset is used as a real regression dataset with a larger number of predictors. For computational tractability, the experiments use a reduced feature set selected from the training data.

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
│   ├── run_regression_pmlb.py
│   ├── run_regression_communities.py
│   ├── create_portfolio_overall_summary.py
│   ├── create_portfolio_final_plots.py
│   ├── create_regression_overall_summary.py
│   ├── create_regression_figures.py
│   └── create_appendix_figures.py
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
pip install numpy pandas scikit-learn matplotlib seaborn yfinance pmlb gurobipy
```

Alternatively, install from the requirements file:

```bash
pip install -r requirements.txt
```

Gurobi also requires a valid academic license.

To check whether Gurobi is working:

```python
import gurobipy as gp

model = gp.Model()
print("Gurobi is working")
```

The Gurobi license file should not be uploaded to GitHub.

---

## 6. Experiments

### 6.1 Portfolio Experiments

The portfolio experiments include Yahoo Finance test cases and OR-Library benchmark instances.

The workflow is:

1. Load or download portfolio data
2. Calculate returns, expected returns, and covariance matrices where needed
3. Solve the no-sparsity mean-variance portfolio model
4. Solve the sparse portfolio problem using Gurobi
5. Solve the sparse portfolio problem using Genetic Algorithm
6. Solve the sparse portfolio problem using Simulated Annealing
7. Compare the results across methods and values of K

The main comparison metrics are:

* Portfolio variance
* Portfolio risk
* Expected return
* Number of selected assets
* Runtime
* Gurobi optimality gap
* Solver status

To run the OR-Library portfolio experiments:

```bash
python experiments/run_portfolio_orlibrary.py
```

To run the Yahoo Finance portfolio test:

```bash
python experiments/run_portfolio_yahoo_test.py
```

To create portfolio summary tables and figures:

```bash
python experiments/create_portfolio_overall_summary.py
python experiments/create_portfolio_final_plots.py
```

### 6.2 Regression Experiments

The regression experiments compare sparse and non-sparse regression methods on the selected datasets.

The workflow is:

1. Load the dataset
2. Preprocess and standardize features
3. Split the data into training and testing sets
4. Fit ordinary least squares without sparsity
5. Fit LASSO
6. Solve best subset selection using Gurobi MIQP
7. Fit Iterative Hard Thresholding
8. Compare prediction performance and sparsity

The main comparison metrics are:

* Training MSE
* Test MSE
* Number of selected features
* Selected feature names
* Runtime where available
* Gurobi solver status

To run the regression experiments:

```bash
python experiments/run_regression_diabetes.py
python experiments/run_regression_pmlb.py
python experiments/run_regression_communities.py
```

To create regression summary tables and figures:

```bash
python experiments/create_regression_overall_summary.py
python experiments/create_regression_figures.py
```

---

## 7. Results

The generated results are stored in:

```text
results/tables/
results/figures/
```

The tables include portfolio and regression summary results, while the figures are used to visualize the comparison between methods.

Examples of generated outputs include:

* Portfolio variance by dataset, K, and method
* Number of selected assets by dataset, K, and method
* Portfolio runtime heatmap
* Regression test MSE comparison
* Number of selected regression features
* Appendix figures for additional comparison

---

## 8. Literature Connection

The portfolio part is related to cardinality-constrained portfolio optimization and sparse portfolio selection.

The regression part is related to best subset selection, LASSO-type methods, and hard-thresholding methods.

The selected papers are used as representative examples of different method families rather than as a complete review of all work in these areas.

Main references connected to the computational methods include:

* Portfolio Selection
* Heuristics for Cardinality Constrained Portfolio Optimisation
* A Scalable Algorithm for Sparse Portfolio Selection
* Best Subset Selection via a Modern Optimization Lens
* Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms
* An Alternating Method for Cardinality-Constrained Optimization
* Sensitivity of L1 Minimization to Parameter Choice
* The Trimmed Lasso
* Fast Iterative Hard Thresholding Methods with Pruning Gradient Computations
* Cardinality Minimization, Constraints, and Regularization: A Survey

---

## 9. Notes

Some experiments require Gurobi, so results may depend on whether a valid Gurobi license is available.

Yahoo Finance data may change depending on the download date and data availability.

The Gurobi license file is not included in this repository and should not be committed.

---

## 10. Dissertation Context

This code supports the MSc dissertation:

**Optimization Problems with Hard Cardinality Constraints: A Study of Approximate Solution Methods**

The code is intended to provide a reproducible computational basis for the portfolio optimization and sparse regression experiments discussed in the dissertation.