import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_dataset(filepath: str, seed: int = 42):

    df = pd.read_csv(filepath)

    # The last column is always the label
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values

    # Encode labels as integers 0, 1, 2, ...
    # This matters if labels are strings or non-zero-indexed
    _, y = np.unique(y, return_inverse=True)

    n_features = X.shape[1]

    # stratify=y keeps class ratios the same in train and test
    # random_state makes the split reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, n_features


def normalize(X_train: np.ndarray, X_test: np.ndarray):

    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)   # fit + transform on train
    X_test_norm  = scaler.transform(X_test)        # transform only on test

    return X_train_norm, X_test_norm