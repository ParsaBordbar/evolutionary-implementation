import pandas as pd
from configs import cfg
from ga import simple_GA_pipeline
from plot import analyze_and_plot_ga_results
import numpy as np


def set_global_seed(seed=cfg.random_seed):
    np.random.seed(seed)


def run_experiments():
    mutation_probs = [0.2, 0.5, 1.0]
    crossover_probs = [0.5, 1.0]
    modes = ["cutfill", "pmx", "multi"]
    cuts = [1, 2, 3]
    mutations = ['bitwise', 'swap']
    elitisms = [False, True]

    all_results = []

    for mutation in mutations:
            for mut in mutation_probs:
                for cross in crossover_probs:
                    for mode in modes:
                        for elite in elitisms:

                            if mode == "multi":
                                cut_list = cuts
                            else:
                                cut_list = [1]  # Default single-cut for others

                            for cut in cut_list:
                                cfg.mutation_probability = mut
                                cfg.crossover_probability = cross

                                print(f"\n--- Run: mut={mut}, cross={cross}, mode={mode}, cuts={cut}, elitism={elite} ---")

                                _, df = simple_GA_pipeline(crossover_mode=mode, elitism=elite, cuts=cut)

                                final_fit = df["best_fitness"].iloc[-1]
                                gens = df["generation"].iloc[-1]

                                all_results.append({
                                    "mutation_prob": mut,
                                    "mutationType": mutation,
                                    "crossover_prob": cross,
                                    "mode": mode,
                                    "cuts": cut,
                                    "elitism": elite,
                                    "final_best_fitness": final_fit,
                                    "generations": gens,
                                })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("ga_experiment_results.csv", index=False)
    print("\n✅ Experiment Results Saved to ga_experiment_results.csv")
    print(results_df)
    return results_df


def main():
    set_global_seed()
# some basic tests uncomment to run
    #simple_GA_pipeline(crossover_mode="cutfill")
    #simple_GA_pipeline(crossover_mode="pmx")
    run_experiments()
    analyze_and_plot_ga_results("ga_experiment_results.csv")


if __name__ == "__main__":
    main()
