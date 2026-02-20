"""
knn.py  –  k-Nearest Neighbours classifier (no ML libraries)

Key design choices:
  - Fully vectorized: all pairwise distances in one matrix op
  - Accepts a binary mask so we only compute distances on
    the SELECTED features (this is the core of feature selection)
  - Returns both predictions and error rate for convenience
"""

import numpy as np


def _pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distances between all rows of A and B.

    Parameters
    ----------
    A : shape (m, d)
    B : shape (n, d)

    Returns
    -------
    D : shape (m, n)  where D[i, j] = ||A[i] - B[j]||²

    Math trick (avoids an explicit loop):
        ||a - b||² = ||a||² + ||b||² - 2·(a @ b.T)
    This computes all m×n distances in pure NumPy (very fast).
    """
    # (m,)  squared norms of each row in A
    A_sq = np.einsum('ij,ij->i', A, A)   # equivalent to (A**2).sum(axis=1)
    # (n,)  squared norms of each row in B
    B_sq = np.einsum('ij,ij->i', B, B)

    # (m, n)  cross term
    cross = A @ B.T   # shape (m, n)

    # Broadcasting: (m,1) + (n,) - 2*(m,n)  →  (m, n)
    D_sq = A_sq[:, None] + B_sq[None, :] - 2.0 * cross

    # Clip small negatives caused by floating point errors
    np.clip(D_sq, 0.0, None, out=D_sq)

    return D_sq   # we don't need sqrt — order is the same


def _majority_vote(labels: np.ndarray) -> int:
    """Return the most common label. Ties broken by smallest label index."""
    counts = np.bincount(labels)
    return int(np.argmax(counts))


def knn_predict(X_train: np.ndarray,
                y_train: np.ndarray,
                X_test:  np.ndarray,
                k:       int = 3) -> np.ndarray:
    """
    Predict labels for X_test using k-NN trained on X_train/y_train.

    Parameters
    ----------
    X_train : (n_train, d)
    y_train : (n_train,)   integer labels
    X_test  : (n_test,  d)
    k       : number of neighbours

    Returns
    -------
    y_pred  : (n_test,)   integer predicted labels
    """
    # All pairwise squared distances: shape (n_test, n_train)
    D = _pairwise_distances(X_test, X_train)

    # For each test point, get the k nearest training indices
    # argpartition is O(n) vs O(n log n) for argsort — faster!
    # It guarantees the k smallest are in the first k positions
    # (not necessarily sorted among themselves, but that's fine)
    k = min(k, len(y_train))   # guard for tiny datasets
    nn_indices = np.argpartition(D, k, axis=1)[:, :k]  # (n_test, k)

    # Look up their labels and take majority vote
    nn_labels = y_train[nn_indices]   # (n_test, k)
    y_pred = np.array([_majority_vote(row) for row in nn_labels])

    return y_pred


def knn_error_rate(X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test:  np.ndarray,
                   y_test:  np.ndarray,
                   mask:    np.ndarray,
                   k:       int = 3) -> float:
    """
    Compute classification error (%) on selected features only.

    Parameters
    ----------
    mask : binary array of shape (n_features,)
           1 = feature selected, 0 = feature ignored

    Returns
    -------
    error : float in [0, 100]
            (NError / NAll) × 100  — exactly Equation (2) in the paper
    """
    selected = mask.astype(bool)

    # Edge case: if no features selected, predict majority class
    if selected.sum() == 0:
        majority = int(np.bincount(y_train).argmax())
        errors = np.sum(y_test != majority)
        return (errors / len(y_test)) * 100.0

    X_tr = X_train[:, selected]
    X_te = X_test[:, selected]

    y_pred = knn_predict(X_tr, y_train, X_te, k=k)
    errors = np.sum(y_pred != y_test)
    return (errors / len(y_test)) * 100.0


def knn_cross_val_error(X: np.ndarray,
                        y: np.ndarray,
                        mask: np.ndarray,
                        k: int = 3,
                        n_folds: int = 3) -> float:
    """
    n-fold cross-validation error on selected features.

    This is what the paper uses for fitness evaluation during evolution:
        k=3, n_folds=3

    sklearn's KFold is used here only for generating clean fold indices —
    the actual k-NN prediction is entirely our own implementation.
    Using KFold avoids off-by-one bugs in manual index arithmetic,
    and handles remainders correctly when n % n_folds != 0.
    """
    from sklearn.model_selection import KFold

    selected = mask.astype(bool)
    n = len(y)

    # Edge case: no features selected → predict majority class
    if selected.sum() == 0:
        majority = int(np.bincount(y).argmax())
        return (np.sum(y != majority) / n) * 100.0

    X_sel = X[:, selected]

    # shuffle=False: data was already shuffled at load time
    kf = KFold(n_splits=n_folds, shuffle=False)
    total_error = 0.0

    for train_idx, val_idx in kf.split(X_sel):
        X_tr, y_tr = X_sel[train_idx], y[train_idx]
        X_va, y_va = X_sel[val_idx],   y[val_idx]

        y_pred = knn_predict(X_tr, y_tr, X_va, k=k)
        errors = np.sum(y_pred != y_va)
        total_error += (errors / len(y_va)) * 100.0

    return total_error / n_folds