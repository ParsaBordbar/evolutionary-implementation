import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path)
    return df


def stratified_split(df, target_col, test_size=0.3, random_state=42):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def standardize_train_test(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    X_train_scaled = (X_train - mu) / sigma
    X_test_scaled = (X_test - mu) / sigma
    return X_train_scaled, X_test_scaled, mu, sigma