from configs import MAX_DEPTH, MAX_GEN, MAX_STEPS, MAZE, POP_SIZE, START, GOAL
from tree_nodes import generate_tree
from genome import Individual
from gp_operators import evaluate, select, crossover, mutate
import matplotlib.pyplot as plt

from utils import visualize_maze



def evolve():
    population = [Individual(generate_tree(MAX_DEPTH)) for _ in range(POP_SIZE)]

    best_hist = []
    avg_hist = []

    for gen in range(MAX_GEN):
        for ind in population:
            evaluate(ind, MAZE, START, GOAL, MAX_STEPS)

        population.sort(key=lambda x: x.fitness)

        best = population[0].fitness
        avg = sum(i.fitness for i in population)/POP_SIZE

        best_hist.append(best)
        avg_hist.append(avg)

        print(f"Gen {gen} | Best: {best} | Avg: {avg:.2f}")

        if best == 0:
            print("Solution found!")
            print(population[0].tree)
            break

        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1 = select(population)
            p2 = select(population)
            child = crossover(p1, p2)
            mutate(child, MAX_DEPTH)
            new_pop.append(child)

        population = new_pop
        
    best = population[0]
    agent = evaluate(best, MAZE, START, GOAL, MAX_STEPS, return_agent=True)
    print(agent.path)

    visualize_maze(MAZE, START, GOAL)


    plt.plot(best_hist, label="Best Fitness")
    plt.plot(avg_hist, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GP Maze Solver Evolution")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    evolve()