
# Set of Utility Functions
import numpy as np
from configs import cfg


def swap(item1, item2):
    return item2, item1

def select_a_random_chromosome(N=cfg.n_queens):
    return np.random.randint(0, N - 1)

def select_a_random_phenotype(population_size):
    return np.random.randint(0, population_size - 1)

def generate_chromosome(N=cfg.n_queens):
   return np.random.permutation(N)

def log_generation(generation, mean_fitness, best_fitness, config):
    data = {
        "generation": generation,
        "mean_fitness": mean_fitness,
        "best_fitness": best_fitness,
        "mutation_prob": config.mutation_probability,
        "crossover_prob": config.crossover_probability,
        "mutation_type": config.mutation_type,
        "n_queens": config.n_queens
    }
    return data

# This utility function checks for mutations that are wrong and fix them (For bitwise mutations to preserve permutation)
def repair_child(child, N):
    unique_genes = set(child)
    missing_genes = [g for g in range(N) if g not in unique_genes]

    seen = set()
    for i, gene in enumerate(child):
        if gene in seen:
            child[i] = missing_genes.pop()
        else:
            seen.add(gene)
    return child

def swap_indices(arr, gene_a, gene_b):
    idx_a, idx_b = np.where(arr == gene_a)[0], np.where(arr == gene_b)[0]
    if idx_a.size and idx_b.size:
        arr[idx_a[0]], arr[idx_b[0]] = arr[idx_b[0]], arr[idx_a[0]]
