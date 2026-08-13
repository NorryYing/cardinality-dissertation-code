"""Data-loading utilities for sparse regression experiments.

The diabetes dataset is used as a small prototype dataset for developing and
testing sparse regression methods before scaling experiments to larger
benchmark datasets.
"""

from __future__ import annotations

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import pmlb
from ucimlrepo import fetch_ucirepo


def load_diabetes_data(test_size=0.2, random_state=42, standardize=True):
	"""Load and split the sklearn diabetes regression dataset.

	Parameters
	----------
	test_size : float or int, optional
		Fraction or number of observations assigned to the test set.
	random_state : int, optional
		Seed used for the reproducible train/test split.
	standardize : bool, optional
		If true, fit a ``StandardScaler`` on the training features and apply
		it to both the training and test features. The response remains on its
		original scale.

	Returns
	-------
	tuple
		``X_train, X_test, y_train, y_test, feature_names``.
	"""
	# This compact dataset is the small prototype dataset for sparse
	# regression experiments and algorithm development.
	dataset = load_diabetes()
	X = dataset.data
	y = dataset.target
	feature_names = list(dataset.feature_names)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
	)

	if standardize:
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

	return X_train, X_test, y_train, y_test, feature_names


def load_pmlb_regression_data(dataset_name="197_cpu_act", test_size=0.2, random_state=42, standardize=True):
	"""Load and split a PMLB regression dataset.

	The PMLB (Penn Machine Learning Benchmarks) repository provides curated
	regression datasets of varying complexity. The 197_cpu_act dataset is a
	medium-size regression benchmark used for testing sparse regression
	methods after prototyping on the small Diabetes dataset.

	Parameters
	----------
	dataset_name : str, optional
		Name of the PMLB dataset to load (default: "197_cpu_act").
	test_size : float or int, optional
		Fraction or number of observations assigned to the test set.
	random_state : int, optional
		Seed used for the reproducible train/test split.
	standardize : bool, optional
		If true, fit a ``StandardScaler`` on the training features and apply
		it to both the training and test features. The response remains on its
		original scale.

	Returns
	-------
	tuple
		``X_train, X_test, y_train, y_test, feature_names``.

	Raises
	------
	RuntimeError
		If the dataset cannot be downloaded or loaded.
	"""
	try:
		# Fetch the dataset from PMLB
		df = pmlb.fetch_data(dataset_name)
	except Exception as e:
		raise RuntimeError(
			f"Failed to load PMLB dataset '{dataset_name}'. "
			f"Please ensure the dataset exists and internet connectivity is available. "
			f"Error details: {e}"
		)

	try:
		# Separate features and target
		# The target column is named "target" in PMLB datasets
		if "target" not in df.columns:
			raise ValueError(
				f"Dataset '{dataset_name}' does not have a 'target' column. "
				f"Available columns: {list(df.columns)}"
			)

		y = df["target"].values
		X = df.drop("target", axis=1).values
		feature_names = list(df.drop("target", axis=1).columns)

	except Exception as e:
		raise RuntimeError(
			f"Failed to extract features and target from dataset '{dataset_name}'. "
			f"Error details: {e}"
		)

	# Split into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
	)

	# Standardize features if requested
	if standardize:
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

	return X_train, X_test, y_train, y_test, feature_names


def load_communities_crime_data(test_size=0.2, random_state=42, standardize=True):
	"""Load and split the UCI Communities and Crime regression dataset.

	The Communities and Crime dataset is a large, real-world regression dataset
	from the UCI Machine Learning Repository (ID: 183). It contains 1,994 samples
	with 128 features representing demographic and policing information for
	communities in the United States. The task is to predict violent crime rates
	per population (ViolentCrimesPerPop).

	This dataset is used for testing sparse regression methods on a realistic
	high-dimensional problem. Missing values are imputed using the median
	strategy, and non-numeric identifier columns are removed as they are not
	meaningful regression predictors.

	Parameters
	----------
	test_size : float or int, optional
		Fraction or number of observations assigned to the test set.
	random_state : int, optional
		Seed used for the reproducible train/test split.
	standardize : bool, optional
		If true, fit a ``StandardScaler`` on the training features and apply
		it to both the training and test features. The response remains on its
		original scale.

	Returns
	-------
	tuple
		``X_train, X_test, y_train, y_test, feature_names``.

	Raises
	------
	RuntimeError
		If the dataset cannot be downloaded or loaded from the UCI repository.
	"""
	try:
		# Fetch the dataset from UCI repository
		communities = fetch_ucirepo(id=183)
	except Exception as e:
		raise RuntimeError(
			f"Failed to load UCI Communities and Crime dataset (ID: 183). "
			f"Please ensure internet connectivity is available. "
			f"Error details: {e}"
		)

	try:
		# Extract features and target
		X = communities.data.features
		y = communities.data.targets

		# Handle target: if it's a DataFrame, extract the ViolentCrimesPerPop column
		if isinstance(y, pd.DataFrame):
			if "ViolentCrimesPerPop" in y.columns:
				y = y["ViolentCrimesPerPop"].values
			else:
				raise ValueError(
					f"Target DataFrame does not have 'ViolentCrimesPerPop' column. "
					f"Available columns: {list(y.columns)}"
				)
		else:
			y = np.asarray(y).flatten()

		# Remove non-numeric identifier columns that are not meaningful predictors
		identifier_cols = ["communityname", "state", "county", "community", "fold"]
		cols_to_drop = [col for col in identifier_cols if col in X.columns]
		if cols_to_drop:
			X = X.drop(columns=cols_to_drop)

		# Convert all columns to numeric, handling non-numeric values
		for col in X.columns:
			X[col] = pd.to_numeric(X[col], errors="coerce")

		# Impute missing values using median strategy
		# This preserves the dataset size while handling missing data
		imputer = SimpleImputer(strategy="median")
		X_imputed = imputer.fit_transform(X)
		X = pd.DataFrame(X_imputed, columns=X.columns)

		feature_names = list(X.columns)
		X = X.values

	except Exception as e:
		raise RuntimeError(
			f"Failed to process UCI Communities and Crime dataset. "
			f"Error details: {e}"
		)

	# Split into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
	)

	# Standardize features if requested
	if standardize:
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

	return X_train, X_test, y_train, y_test, feature_names
