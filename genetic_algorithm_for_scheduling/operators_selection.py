import numpy as np
from config import TOURNAMENT_K

def tournament_selection(population, fitness_values):
    selected = []
    pop_size = len(population)

    for _ in range(pop_size):
        competitors = np.random.choice(pop_size, TOURNAMENT_K, replace=False)
        best = competitors[np.argmax(fitness_values[competitors])]
        selected.append(population[best].copy())

    return selected