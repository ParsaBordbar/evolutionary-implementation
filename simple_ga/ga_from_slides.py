from pprint import pprint
from ga_from_slides.slides_utils import ga_summary, generate_chromosome, select_a_random_chromosome, slelct_a_random_phenotype, swap

POPULATION_SIZE = 100
PARENET_SELECTION_COUNT = 5
GA_PIPLINE_ROUNDS = 50

def generate_population(size):
    population = []
    for _ in range(size):
        population.append(generate_chromosome())
    return population

def fitness_evaluation(queens):
    penalty = 0
    n = len(queens)
    
    for i in range(n):
        for j in range(i + 1, n):
            if queens[i] == queens[j]:
                penalty += 1
            elif abs(queens[i] - queens[j]) == abs(i - j):
                penalty += 1
                
    # Fitness is inverse of penalty, added 1 to avoid division by zero
    fitness = (1 / (1 + penalty))
    return fitness.__round__(2)


def parent_selection(selection_count, population=generate_population(POPULATION_SIZE)):
    sample = []
    parents = []

    # Pick 5 random parents
    for _ in range(selection_count):
        random_index = slelct_a_random_phenotype(population_size=len(population)-1)
        sample.append(population[random_index])

    # Evaluatting Fitnesses also we sort them here! first two index will be the best parents
    for i in range(len(sample)):
        fitness = fitness_evaluation(sample[i])
        parents.append({"fitness": fitness, "chromosome": sample[i]})

    return parents[0], parents[1]


def crossover(parent1, parent2):
    if (len(parent1) == 8 or parent2 == 8):
        crossover_point = select_a_random_chromosome()
        if crossover_point < 1: crossover_point =+ 3

        print(f"\nCrossover point at index: {crossover_point} \n")
        
        # Sliccing Chromosome arrays
        parent1_firstpart = parent1[0:crossover_point]
        parent1_secondpart = parent1[(crossover_point):]

        parent2_firstpart = parent2[0:crossover_point]
        parent2_secondpart = parent2[(crossover_point):]
        
        parent2_cycle = parent2_secondpart + parent2_firstpart
        parent1_cycle = parent1_secondpart + parent1_firstpart

        child1_secondpart = []
        for gene in parent2_cycle:
            if gene not in parent1_firstpart:
                child1_secondpart.append(gene)
                if len(parent1_secondpart) == len(child1_secondpart):
                    break
                continue

        print("this is the child1 second part")
        pprint(child1_secondpart)
        child1 = parent1_firstpart + child1_secondpart
        print("\n Child 1", child1, len(child1))

        child2_secondpart = []
        for gene in parent1_cycle:
            if gene not in parent2_firstpart:
                child2_secondpart.append(gene)
                if len(parent2_secondpart) == len(child2_secondpart):
                    break
                continue

        print("\n this is the child2 second part")
        pprint(child1_secondpart)
        child2 = parent2_firstpart + child2_secondpart
        print("\n Child 2", child2, len(child2))

        return [child1, child2]
    else:
        raise ValueError("Chromosome length must be 8")

def mution(queen):
    mut_index_1 = select_a_random_chromosome()
    mut_index_2 = select_a_random_chromosome()

    mut_el1, mut_el2 = swap(queen[mut_index_1], queen[mut_index_2])

    queen[mut_index_1] = mut_el1
    queen[mut_index_2] = mut_el2
    return queen

def survival_selection(population, children ,population_size=POPULATION_SIZE):
    population_fitnesses = []

    # Population's fittness evaluation
    for sample in population:
        fitness = fitness_evaluation(sample)
        population_fitnesses.append({"fitness": fitness, "chromosome": sample})
    population_fitnesses.sort(key=lambda x: x['fitness'], reverse=True)

    # Children's fittness evaluation
    children_fitnesses = []
    for child in children:
        child_fitness = fitness_evaluation(child)
        children_fitnesses.append({"fitness": child_fitness, "chromosome": child})
    print("\n Children's fitnesses: ")
    pprint(children_fitnesses)

    # The Actual Survival Selection
    for i, sample in enumerate(population_fitnesses):
        for j, child in enumerate(children_fitnesses):    
            if sample['fitness'] < children_fitnesses[j]['fitness']:
                population_fitnesses.pop(i)
                population_fitnesses.insert(i, children_fitnesses[0])
                children_fitnesses.pop(j)
                population_fitnesses.sort(key=lambda x: x['fitness'], reverse=True)
    
    return population_fitnesses[:population_size]

def fitness_mean(population):
    total_fitness = 0
    for sample in population:
        total_fitness += fitness_evaluation(sample)
    mean_fitness = total_fitness / len(population)
    return mean_fitness.__round__(4)


def simple_GA_pipline(rounds=GA_PIPLINE_ROUNDS ,population_size=POPULATION_SIZE, parent_selection_count=PARENET_SELECTION_COUNT):
    population = generate_population(population_size)

    for _ in range(rounds):
        parents = parent_selection(parent_selection_count, population)
        crossover_result = crossover(parents[0]['chromosome'], parents[1]['chromosome'])
        mut1 = mution(crossover_result[0])
        mut2 = mution(crossover_result[1])
        children = [mut1, mut2]
        population_fitnesses = survival_selection(population, children, population_size)

        population = []
        for sample in population_fitnesses:
            population.append(sample['chromosome'])
        mean_fitness = fitness_mean(population)
        
        ga_summary(population, parents, crossover_result, children, population, population_fitnesses, mean_fitness)

    return population


def main():
   simple_GA_pipline(10, POPULATION_SIZE, PARENET_SELECTION_COUNT)

if __name__ == "__main__":
    main()
