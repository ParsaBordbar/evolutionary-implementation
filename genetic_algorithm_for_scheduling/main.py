import os
import matplotlib.pyplot as plt
from config import CROSSOVER_METHODS, MUTATION_METHODS, N_JOBS, OUTPUT_DIR
from ga import run_ga


for cross in CROSSOVER_METHODS:
    for mut in MUTATION_METHODS:
        print(f"\n=== Running GA with {cross.upper()} + {mut.upper()} ===")

        best, best_T, max_hist, avg_hist = run_ga(cross, mut, N_JOBS)

        print("Best sequence:", best)
        print("Min total weighted completion time:", best_T)

        plt.figure(figsize=(7,5))
        plt.plot(max_hist, label="Max Fitness")
        plt.plot(avg_hist, label="Avg Fitness")
        plt.title(f"{cross.upper()} + {mut.upper()} Performance")
        plt.legend()
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.grid()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{cross}_{mut}.png"), dpi=300)