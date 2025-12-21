import numpy as np
from utilites import cross_entropy_loss, predict_proba
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


class Individual:
    def __init__(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma
        self.fitness = None


def initialize_population(mu, dim):
    population = []
    for _ in range(mu):
        theta = np.random.uniform(-0.1, 0.1, size=dim)
        sigma = np.random.uniform(0.01, 0.1, size=dim)
        population.append(Individual(theta, sigma))
    return population


def mutate(individual, tau, tau_prime):
    n = len(individual.theta)

    global_noise = np.random.randn()
    local_noise = np.random.randn(n)

    sigma_new = individual.sigma * np.exp(
        tau * global_noise + tau_prime * local_noise
    )
    sigma_new = np.maximum(sigma_new, 1e-6)

    theta_new = individual.theta + sigma_new * np.random.randn(n)
    theta_new = np.clip(theta_new, -5, 5)

    return Individual(theta_new, sigma_new)


def evaluate_population(population, X, y, lambda_reg):
    for ind in population:
        loss = cross_entropy_loss(ind.theta, X, y, lambda_reg)
        ind.fitness = -loss


def select_mu_plus_lambda(parents, offspring, mu):
    combined = parents + offspring
    combined.sort(key=lambda ind: ind.fitness, reverse=True)
    return combined[:mu]


def train_es(
    X_train,
    y_train,
    mu=30,
    lambd=210,
    generations=100,
    lambda_reg=0.01
):
    dim = X_train.shape[1] + 1
    n = dim

    tau = 1 / np.sqrt(2 * n)
    tau_prime = 1 / np.sqrt(2 * np.sqrt(n))

    population = initialize_population(mu, dim)

    best_losses = []
    mean_losses = []
    train_accuracies = []

    for gen in range(generations):
        evaluate_population(population, X_train, y_train, lambda_reg)

        losses = [-ind.fitness for ind in population]
        best_losses.append(np.min(losses))
        mean_losses.append(np.mean(losses))

        best_ind = population[np.argmin(losses)]
        y_pred = (predict_proba(X_train, best_ind.theta) >= 0.5).astype(int)
        train_accuracies.append(accuracy_score(y_train, y_pred))

        offspring = []
        for parent in population:
            for _ in range(lambd // mu):
                offspring.append(mutate(parent, tau, tau_prime))

        evaluate_population(offspring, X_train, y_train, lambda_reg)
        population = select_mu_plus_lambda(population, offspring, mu)

        print(
            f"Gen {gen:03d} | "
            f"Best loss: {best_losses[-1]:.4f} | "
            f"Train acc: {train_accuracies[-1]:.3f}"
        )

    return population, best_losses, mean_losses, train_accuracies

def evaluate_final(individual, X_test, y_test):
    y_prob = predict_proba(X_test, individual.theta)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)

    return acc, precision, recall, f1, cm