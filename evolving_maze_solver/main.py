from configs import MAX_DEPTH, MAX_GEN, MAX_STEPS, MAZE, POP_SIZE, START, GOAL
from tree_nodes import generate_tree
from genome import Individual
from gp_operators import evaluate, select, crossover, mutate
import matplotlib.pyplot as plt
from utils import visualize_maze, print_tree_structure


def evolve():
    """
    Main genetic programming algorithm for maze solving.
    
    Process:
    1. Initialize population using ramped half-and-half
    2. For each generation:
       - Evaluate all individuals
       - Select best parents
       - Create offspring via crossover and mutation
       - Track statistics
    3. Output results and visualizations
    """
    print("="*70)
    print("GENETIC PROGRAMMING MAZE SOLVER")
    print("="*70)
    print(f"Population Size: {POP_SIZE}")
    print(f"Max Generations: {MAX_GEN}")
    print(f"Max Tree Depth: {MAX_DEPTH}")
    print(f"Max Steps per Episode: {MAX_STEPS}")
    print(f"Start: {START}, Goal: {GOAL}")
    print("="*70)
    print()
    
    # Initialize population with ramped half-and-half
    population = []
    print("Initializing population with ramped half-and-half method...")
    for _ in range(POP_SIZE // 2):
        population.append(Individual(generate_tree(MAX_DEPTH, method='grow')))
    for _ in range(POP_SIZE // 2):
        population.append(Individual(generate_tree(MAX_DEPTH, method='full')))
    print(f"✓ Created {len(population)} individuals\n")

    best_hist = []
    avg_hist = []
    worst_hist = []
    std_hist = []

    print("Starting evolution...")
    print(f"{'Gen':<5} {'Best':<10} {'Avg':<10} {'Worst':<10} {'Std':<10} {'Status':<15}")
    print("-" * 70)

    for gen in range(MAX_GEN):
        # Evaluate all individuals
        for ind in population:
            evaluate(ind, MAZE, START, GOAL, MAX_STEPS)

        # Sort by fitness (lower is better)
        population.sort(key=lambda x: x.fitness)

        # Calculate statistics
        fitnesses = [i.fitness for i in population]
        best = fitnesses[0]
        avg = sum(fitnesses) / POP_SIZE
        worst = fitnesses[-1]
        std = (sum((f - avg) ** 2 for f in fitnesses) / POP_SIZE) ** 0.5

        best_hist.append(best)
        avg_hist.append(avg)
        worst_hist.append(worst)
        std_hist.append(std)

        # Status indicator
        if best == 0:
            status = "✓ SOLVED!"
        elif gen % 10 == 0:
            status = "Improving..."
        else:
            status = ""

        print(f"{gen:<5} {best:<10.2f} {avg:<10.2f} {worst:<10.2f} {std:<10.2f} {status:<15}")

        # Termination condition
        if best == 0:
            print("-" * 70)
            print(f"\n✓ Solution found at generation {gen}!\n")
            break

        # Create new population
        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1 = select(population)
            p2 = select(population)
            child = crossover(p1, p2)
            mutate(child, MAX_DEPTH)
            new_pop.append(child)

        population = new_pop

    print("="*70)
    print("EVOLUTION COMPLETE")
    print("="*70)
    print()

    # Get best solution
    best = population[0]
    agent = evaluate(best, MAZE, START, GOAL, MAX_STEPS, return_agent=True)
    
    # Output best solution details
    print("BEST SOLUTION STATISTICS")
    print("="*70)
    print(f"Final Position:        ({agent.x}, {agent.y})")
    print(f"Goal Position:         {GOAL}")
    print(f"Reached Goal:          {'YES ✓' if (agent.x, agent.y) == GOAL else 'NO ✗'}")
    print(f"Steps Taken:           {agent.steps}")
    print(f"Wall Hits:             {agent.wall_hits}")
    print(f"Unique Cells Visited:  {len(agent.visited)}")
    print(f"Loop Count:            {agent.steps - len(agent.visited)}")
    print(f"Fitness Score:         {best.fitness:.2f}")
    print("="*70)
    print()

    # Output solution sequence
    print("EVOLVED SOLUTION - MOVEMENT SEQUENCE")
    print("="*70)
    if len(agent.path) <= 50:
        print(f"Path ({len(agent.path)} steps):")
        for i, pos in enumerate(agent.path):
            print(f"  Step {i:2d}: {pos}")
    else:
        print(f"Path ({len(agent.path)} steps):")
        print(f"  Start: {agent.path[0]}")
        for i in range(1, min(10, len(agent.path))):
            print(f"  Step {i:2d}: {agent.path[i]}")
        print(f"  ...")
        for i in range(max(10, len(agent.path)-5), len(agent.path)):
            print(f"  Step {i:2d}: {agent.path[i]}")
    print("="*70)
    print()

    # Output decision tree
    print("EVOLVED DECISION TREE STRUCTURE")
    print("="*70)
    print_tree_structure(best.tree)
    print("="*70)
    print()

    # Visualization 1: Fitness progression
    print("Generating fitness progression graph...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Best and Average
    ax1.plot(best_hist, label="Best Fitness", linewidth=2.5, marker='o', markersize=4, color='green')
    ax1.plot(avg_hist, label="Average Fitness", linewidth=2.5, marker='s', markersize=4, color='blue')
    ax1.plot(worst_hist, label="Worst Fitness", linewidth=2, linestyle='--', color='red', alpha=0.7)
    ax1.fill_between(range(len(best_hist)), 
                      [avg_hist[i] - std_hist[i] for i in range(len(best_hist))],
                      [avg_hist[i] + std_hist[i] for i in range(len(best_hist))],
                      alpha=0.2, color='blue', label='±1 Std Dev')
    ax1.set_xlabel("Generation", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Fitness Score", fontsize=12, fontweight='bold')
    ax1.set_title("Population Fitness Progression", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Standard Deviation
    ax2.plot(std_hist, label="Population Diversity (Std Dev)", linewidth=2.5, 
             marker='o', markersize=4, color='purple')
    ax2.set_xlabel("Generation", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Standard Deviation", fontsize=12, fontweight='bold')
    ax2.set_title("Population Diversity Over Time", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Visualization 2: Maze with path
    print("Generating maze visualization with solution path...")
    visualize_maze(MAZE, START, GOAL, agent.path)

    # Summary statistics
    print("\nFINAL SUMMARY")
    print("="*70)
    print(f"Total Generations Run:    {len(best_hist)}")
    print(f"Initial Best Fitness:     {best_hist[0]:.2f}")
    print(f"Final Best Fitness:       {best_hist[-1]:.2f}")
    improvement = best_hist[0] - best_hist[-1]
    improvement_pct = (improvement / best_hist[0] * 100) if best_hist[0] != 0 else 0
    print(f"Fitness Improvement:      {improvement:.2f} ({improvement_pct:.1f}%)")
    print(f"Initial Avg Fitness:      {avg_hist[0]:.2f}")
    print(f"Final Avg Fitness:        {avg_hist[-1]:.2f}")
    print("="*70)


if __name__ == "__main__":
    evolve()