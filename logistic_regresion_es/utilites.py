import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X, theta):
    W = theta[:-1]
    b = theta[-1]
    return sigmoid(X @ W + b)


def cross_entropy_loss(theta, X, y, lambda_reg=0.0):
    y_hat = predict_proba(X, theta)
    eps = 1e-8
    ce = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
    l2 = lambda_reg * np.sum(theta[:-1] ** 2)
    return ce + l2

