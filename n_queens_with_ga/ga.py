import numpy as np
import pandas as pd
from configs import cfg
from utils import generate_chromosome, log_generation, repair_child, select_a_random_chromosome


def generate_population(size, N=cfg.n_queens):
    return np.array([generate_chromosome(N) for _ in range(size)])

def fitness_evaluation_vectorized(population):
    n = population.shape[1]
    fitness = np.zeros(len(population))

    for idx, q in enumerate(population):
        # skip invalid chromosomes
        if len(set(q)) != n:
            fitness[idx] = 0.0
            continue

        i, j = np.triu_indices(n, k=1)
        same_row = (q[i] == q[j])
        same_diag = (np.abs(q[i] - q[j]) == np.abs(i - j))
        penalty = np.sum(same_row | same_diag)

        penalty = np.maximum(0, penalty)

        # Handles NaNs
        denom = 1.0 + penalty
        if denom == 0 or np.isnan(denom):
            fitness[idx] = 0.0
        else:
            fitness[idx] = 1.0 / denom

    fitness = np.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0)
    return np.round(fitness, 3)


def parent_selection(population, fitnesses, k=cfg.parent_selection_count):
    population_size = len(population)
    
    sample_indices = np.random.choice(population_size, size=k, replace=False)
    sample_fitnesses = fitnesses[sample_indices]
    sorted_idx = sample_indices[np.argsort(-sample_fitnesses)]
    return population[sorted_idx[0]], population[sorted_idx[1]]

def crossover(parent1, parent2, prob=cfg.crossover_probability, mode="cutfill", cuts=1):
    # Ensure parents are Python lists (not NumPy arrays)
    parent1 = parent1.tolist() if isinstance(parent1, np.ndarray) else parent1
    parent2 = parent2.tolist() if isinstance(parent2, np.ndarray) else parent2

    if np.random.random() > prob:
        return [parent1[:], parent2[:]]

    N = len(parent1)

    # CUT-AND-FILL CROSSOVER
    if mode == "cutfill":
        crossover_point = select_a_random_chromosome(N)
        if crossover_point < 1:
            crossover_point = 3

        p1_first = parent1[:crossover_point]
        p2_cycle = parent2[crossover_point:] + parent2[:crossover_point]

        child1_tail = [g for g in p2_cycle if g not in p1_first][: N - crossover_point]
        child1 = p1_first + child1_tail

        p2_first = parent2[:crossover_point]
        p1_cycle = parent1[crossover_point:] + parent1[:crossover_point]
        child2_tail = [g for g in p1_cycle if g not in p2_first][: N - crossover_point]
        child2 = p2_first + child2_tail

        return [child1, child2]

    # PMX CROSSOVER
    if mode == "pmx":
        c1, c2 = np.sort(np.random.choice(N, size=2, replace=False))
        child1, child2 = parent1[:], parent2[:]

        child1[c1:c2], child2[c1:c2] = parent2[c1:c2], parent1[c1:c2]

        mapping1 = {parent2[i]: parent1[i] for i in range(c1, c2)}
        mapping2 = {parent1[i]: parent2[i] for i in range(c1, c2)}

        def map_gene(gene, mapping):
            while gene in mapping:
                gene = mapping[gene]
            return gene

        for i in list(range(0, c1)) + list(range(c2, N)):
            child1[i] = map_gene(child1[i], mapping1)
            child2[i] = map_gene(child2[i], mapping2)

        return [child1, child2]

    # MULTI-CUT CROSSOVER
    if mode == "multi":
        cuts = min(cuts, 3)
        cut_points = np.sort(np.random.choice(range(1, N - 1), size=cuts, replace=False))
        parts1, parts2 = [], []
        last = 0
        for cp in cut_points + [N]:
            parts1.append(parent1[last:cp])
            parts2.append(parent2[last:cp])
            last = cp

        child1, child2 = [], []
        for i in range(len(parts1)):
            if i % 2 == 0:
                child1 += parts1[i]
                child2 += parts2[i]
            else:
                child1 += parts2[i]
                child2 += parts1[i]

        return [child1, child2]

    return [parent1[:], parent2[:]]

def mutation(chromosome, prob=cfg.mutation_probability, mode=cfg.mutation_type):
    if np.random.random() > prob:
        return chromosome

    N = len(chromosome)
    i, j = select_a_random_chromosome(N), select_a_random_chromosome(N)

    if mode == "swap":
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

    elif mode == "bitwise":
        chromosome[i] = np.random.choice([x for x in range(N) if x not in chromosome or x == chromosome[i]])
        repaired_chromosome = repair_child(chromosome, N)
        return repaired_chromosome


def survival_selection(population, children, population_size=cfg.population_size, elitism=False, elite_count=2):
    all_samples = population + children
    fitnesses = [{"fitness": fitness_evaluation_vectorized(s), "chromosome": s} for s in all_samples]
    fitnesses.sort(key=lambda x: x['fitness'], reverse=True)

    if elitism:
        elites = fitnesses[:elite_count]
        remaining = fitnesses[elite_count:]
        next_gen = elites + remaining[: population_size - elite_count]
        return [f["chromosome"] for f in next_gen]
    else:
        return [f["chromosome"] for f in fitnesses[:population_size]]

def fitness_mean(fitnesses):
    return np.mean(fitnesses)

# This Function Combines everything above it as a single pipeline! 
def simple_GA_pipeline(crossover_mode="cutfill", seed=42, cuts=1, elitism=False, mutationType=cfg.mutation_type):
    population = generate_population(cfg.population_size, cfg.n_queens)
    fitnesses = fitness_evaluation_vectorized(population)
    log_data = []

    for gen in range(cfg.ga_pipeline_rounds):
        parent1, parent2 = parent_selection(population, fitnesses)

        children = np.array(crossover(parent1, parent2, mode=crossover_mode, cuts=cuts))
        children = np.array([mutation(c, cfg.mutation_probability, mutationType) for c in children])

        all_samples = np.vstack((population, children))
        all_fitness = fitness_evaluation_vectorized(all_samples)
        sorted_idx = np.argsort(-all_fitness)
        population = all_samples[sorted_idx[:cfg.population_size]]
        fitnesses = all_fitness[sorted_idx[:cfg.population_size]]

        # logger
        mean_fit = np.mean(fitnesses)
        best_fit = np.max(fitnesses)
        log_data.append(log_generation(gen, mean_fit, best_fit, cfg))

        if best_fit == 1:
            print(f"✅ Found solution in {gen} generations")
            break

    df_log = pd.DataFrame(log_data)
    return population, df_log
