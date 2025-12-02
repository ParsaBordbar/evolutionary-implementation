import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_and_plot_ga_results(csv_path="ga_experiment_results.csv", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    df["success"] = df["final_best_fitness"] == 1.0
    success_rate = df["success"].mean() * 100

    # Average generations grouped
    avg_generations = df.groupby(
        ["mutation_prob", "mode", "elitism"]
    )["generations"].mean().reset_index()

    # Plot 1: Convergence Speed by Mutation Rate and Mode (line plot)
    plt.figure(figsize=(8,5))

    modes = df["mode"].unique()
    elitism_values = df["elitism"].unique()

    markers = ["o", "s", "D", "^", "v", ">"]
    marker_idx = 0

    # Plot lines manually
    for mode in modes:
        for elite in elitism_values:
            sub = avg_generations[(avg_generations["mode"] == mode) &
                                  (avg_generations["elitism"] == elite)]

            plt.plot(
                sub["mutation_prob"],
                sub["generations"],
                marker=markers[marker_idx % len(markers)],
                label=f"{mode}, elitism={elite}"
            )
            marker_idx += 1

    plt.title("Average Convergence Speed by Mutation & Mode")
    plt.xlabel("Mutation Probability")
    plt.ylabel("Average Generations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speed_by_mutation_mode.png"))
    plt.close()

    # Plot 2: Elitism Effect (simple bar chart)
    plt.figure(figsize=(6,4))

    avg_elitism = df.groupby("elitism")["generations"].mean()

    plt.bar(
        ["No Elitism", "Elitism"],
        avg_elitism.values,
        color=["gray", "steelblue"]
    )

    plt.title("Effect of Elitism on Convergence Speed")
    plt.ylabel("Average Generations")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "elitism_effect.png"))
    plt.close()

    # Plot 3: Distribution by Mode (boxplot)
    plt.figure(figsize=(8,5))

    modes = df["mode"].unique()
    data = [df[df["mode"] == m]["generations"].values for m in modes]

    plt.boxplot(data, labels=modes)

    plt.title("Distribution of Generations by Crossover Mode")
    plt.xlabel("Crossover Mode")
    plt.ylabel("Generations")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_by_mode.png"))
    plt.close()

    # Plot 4: Multi-Cut Performance
    multi_df = df[df["mode"] == "multi"]

    if not multi_df.empty:
        plt.figure(figsize=(8,5))

        cuts = sorted(multi_df["cuts"].unique())
        elitism_values = multi_df["elitism"].unique()

        width = 0.35
        x = range(len(cuts))

        for i, elite in enumerate(elitism_values):
            sub = multi_df[multi_df["elitism"] == elite]
            means = [sub[sub["cuts"] == c]["generations"].mean() for c in cuts]

            plt.bar(
                [xi + (i*width) for xi in x],
                means,
                width=width,
                label=f"elitism={elite}"
            )

        plt.xticks([xi + width/2 for xi in x], cuts)
        plt.xlabel("Number of Cuts")
        plt.ylabel("Avg Generations")
        plt.title("Multi-Cut Crossover Performance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "multi_cut_performance.png"))
        plt.close()

    # Summary
    print("✅ Genetic Algorithm Analysis Summary")
    print(f"Success Rate: {success_rate:.2f}% ({df['success'].sum()}/{len(df)}) cases solved.")

    best_config = df[df["final_best_fitness"] == 1.0].sort_values("generations").iloc[0]

    print("\nFastest configuration:")
    print(best_config[[
        "mutation_prob", "mode", "elitism", "crossover_prob", "generations"
    ]].to_string(index=False))

    print("\nPlots saved in:", output_dir)
