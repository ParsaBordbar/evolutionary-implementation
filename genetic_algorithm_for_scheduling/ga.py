import numpy as np
from problem import fitness
from operators_selection import tournament_selection
from operators_crossover import order_crossover, pmx, cycle_crossover
from operators_mutation import swap_mutation, inversion_mutation, scramble_mutation

from config import POP_SIZE, MAX_GEN, ELITISM

CROSS = {
    "order": order_crossover,
    "pmx": pmx,
    "cycle": cycle_crossover
}

MUT = {
    "swap": swap_mutation,
    "inversion": inversion_mutation,
    "scramble": scramble_mutation
}


def run_ga(crossover_name, mutation_name, n_jobs):
    population = np.array([np.random.permutation(n_jobs) for _ in range(POP_SIZE)])
    max_fit_history, avg_fit_history = [], []

    for _ in range(MAX_GEN):
        fit_values = np.array([fitness(ind) for ind in population])
        max_fit_history.append(fit_values.max())
        avg_fit_history.append(fit_values.mean())

        elite_idx = np.argsort(fit_values)[-ELITISM:]
        elites = population[elite_idx]

        mating_pool = tournament_selection(population, fit_values)

        children = []
        for i in range(0, POP_SIZE - ELITISM, 2):
            p1, p2 = mating_pool[i], mating_pool[i + 1]
            c1 = CROSS[crossover_name](p1, p2)
            c2 = CROSS[crossover_name](p2, p1)
            children.append(MUT[mutation_name](c1))
            children.append(MUT[mutation_name](c2))

        population = np.vstack((elites, np.array(children)))

    best_idx = np.argmax([fitness(ind) for ind in population])
    best = population[best_idx]
    best_T = 1 / fitness(best)

    return best, best_T, max_fit_history, avg_fit_history