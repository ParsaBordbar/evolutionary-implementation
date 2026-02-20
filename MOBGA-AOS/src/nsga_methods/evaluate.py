"""
evaluate.py  –  Fitness evaluation wrapper for MOBGA-AOS

This module defines the two objective functions from the paper:

    f1 = classification error (%) via k-NN (k=3) + 3-fold CV  → minimize
    f2 = number of selected features (sum of binary mask)       → minimize

These map directly to equations (2) and (3) in the paper.

Why a separate file?
  mobga_aos.py calls _evaluate() internally with caching.
  This module provides the raw uncached versions so they can be
  tested independently and reused for test-set evaluation at the end.
"""

import numpy as np
from knn import knn_cross_val_error, knn_error_rate


def evaluate_individual(
    individual: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 3,
    n_folds: int = 3,
) -> tuple[float, float]:
    """
    Compute both fitness objectives for one individual.

    Used during evolution (training set + cross-validation).

    Parameters
    ----------
    individual : binary array shape (D,)  — 1 = feature selected
    X_train    : shape (n_train, D)
    y_train    : shape (n_train,)
    k          : k-NN neighbours (paper uses 3)
    n_folds    : cross-validation folds (paper uses 3)

    Returns
    -------
    f1 : classification error % in [0, 100]   (lower is better)
    f2 : number of selected features           (lower is better)
    """
    f2 = float(individual.sum())

    if f2 == 0:
        # No features selected → penalize with worst possible error
        # Don't call k-NN since there's nothing to classify with
        return 100.0, 0.0

    f1 = knn_cross_val_error(X_train, y_train, individual,
                              k=k, n_folds=n_folds)
    return f1, f2


def evaluate_on_test(
    individual: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k: int = 3,
) -> tuple[float, float]:
    """
    Evaluate a solution on the held-out TEST set.

    This is only called AFTER evolution ends — never during it.
    Training data is still used to fit the k-NN; test data is
    only used for measuring the final error.

    Returns
    -------
    f1_test : test classification error %
    f2      : number of selected features (same as training)
    """
    f2 = float(individual.sum())

    if f2 == 0:
        return 100.0, 0.0

    f1_test = knn_error_rate(X_train, y_train, X_test, y_test,
                              individual, k=k)
    return f1_test, f2