from pprint import pprint
import random

def swap(item1, item2):
    item1 = item1 ^ item2
    item2 = item1 ^ item2
    item1 = item1 ^ item2
    return item1, item2

def select_a_random_chromosome():
    index = random.randint(0, 7)
    return index

def slelct_a_random_phenotype(population_size):
    index = random.randint(0, population_size)
    return index

def generate_chromosome():
    return random.sample(range(1, 9), 8)

def ga_summary(original_population, parents, crossover_result, mutated_children, new_population, survival_selection, mean_fitness):
    print("\n----- GA Summary -----")
    print(f"Original Population Size: {len(original_population)}")
    print(f"Selected Parents:")
    pprint(parents)
    print(f"Crossover Result:")
    pprint(crossover_result)
    pprint(f"Mutated Children:")
    pprint(mutated_children)
    pprint(f"New Population Size: {len(new_population)}")
    pprint(f"Survival_selection: ")
    pprint(survival_selection)
    pprint(f"Mean Fitness: {mean_fitness}")
    pprint("----------------------")
