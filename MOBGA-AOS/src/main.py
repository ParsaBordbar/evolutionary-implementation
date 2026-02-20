"""
run.py  –  Entry point for MOBGA-AOS experiments

Usage:
    python run.py                    # runs on all datasets with default settings
    python run.py --dataset DS02     # single dataset
    python run.py --max_fes 50000    # smaller budget for quick testing
    python run.py --runs 5           # fewer runs for speed
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from configs.configs import DATA_PATH
from data_loader.data import load_dataset, normalize
from mobga_aos import MOBGA_AOS
from metrics import compute_igd, compute_hv, get_reference_point, build_true_pareto_front
from nsga_methods.crossover import OPERATOR_NAMES


# ── Dataset registry ─────────────────────────────────────────────────────────

DATASETS = {
    "DS02": "DS02.csv",
    "DS04": "DS04.csv",
    "DS05": "DS05.csv",
    "DS07": "DS07.csv",
    "DS08": "DS08.csv",
    "DS10": "DS10.csv",
}


def find_dataset(name: str, data_dir: str = DATA_PATH ) -> str:
    """Search for dataset file in common locations."""
    candidates = [
        f"{data_dir}/{DATASETS[name]}",
        f"{DATA_PATH}/{DATASETS[name]}",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    raise FileNotFoundError(f"Could not find {name} in {candidates}")


# ── Single run ────────────────────────────────────────────────────────────────

def run_once(X_train, y_train, n_features, max_fes, seed, verbose=False):
    """Run MOBGA-AOS once and return the Pareto front objectives."""
    algo = MOBGA_AOS(
        X_train=X_train,
        y_train=y_train,
        n_features=n_features,
        max_fes=max_fes,
        pop_size=100,
        crossover_rate=0.9,
        Q=5,
        LP=5,
        seed=seed,
        verbose=verbose,
    )
    solutions, objectives = algo.run()
    return solutions, objectives, algo.aos.probs   # also return final operator probs


# ── Experiment on one dataset ─────────────────────────────────────────────────

def run_experiment(dataset_name: str,
                   data_dir: str  = ".",
                   n_runs: int    = 3,
                   max_fes: int   = 300_000,
                   verbose: bool  = True):
    """
    Run MOBGA-AOS n_runs times on one dataset, report IGD/HV, plot Pareto front.
    """
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}   runs={n_runs}   maxFEs={max_fes}")
    print(f"{'='*60}")

    # Load and normalize
    path = find_dataset(dataset_name, data_dir)
    X_train, X_test, y_train, y_test, n_features = load_dataset(path)
    X_train, X_test = normalize(X_train, X_test)

    print(f"  Features: {n_features}  |  Train: {len(y_train)}  |  Test: {len(y_test)}")

    all_train_obj = []
    all_op_probs  = []

    for run in range(n_runs):
        t0 = time.perf_counter()
        print(f"\n  ── Run {run+1}/{n_runs} (seed={run}) ──")

        sols, obj, op_probs = run_once(
            X_train, y_train, n_features,
            max_fes=max_fes, seed=run,
            verbose=verbose
        )
        elapsed = time.perf_counter() - t0

        all_train_obj.append(obj)
        all_op_probs.append(op_probs)

        print(f"  Time: {elapsed:.1f}s  |  Pareto solutions: {len(obj)}")
        print(f"  Best train error: {obj[:,0].min():.3f}%  |  Fewest features: {obj[:,1].min():.0f}")

    # ── Build "true" Pareto front from union of all runs ──
    true_pf = build_true_pareto_front(all_train_obj)

    # ── Compute IGD and HV for each run ──
    ref_point = get_reference_point(all_train_obj, margin=1.1)

    igd_scores = []
    hv_scores  = []
    for obj in all_train_obj:
        igd_scores.append(compute_igd(obj, true_pf))
        hv_scores.append(compute_hv(obj, ref_point))

    print(f"\n  ── Summary over {n_runs} runs ──")
    print(f"  IGD:  mean={np.mean(igd_scores):.4f}  std={np.std(igd_scores):.4f}")
    print(f"  HV:   mean={np.mean(hv_scores):.4f}   std={np.std(hv_scores):.4f}")

    avg_probs = np.mean(all_op_probs, axis=0)
    print(f"\n  Final operator probabilities (averaged over runs):")
    for name, prob in zip(OPERATOR_NAMES, avg_probs):
        bar = "█" * int(prob * 40)
        print(f"    {name:20s} {prob:.3f}  {bar}")

    # ── Plot ──
    plot_pareto_front(dataset_name, all_train_obj, true_pf, n_features)

    return {
        "igd_mean": np.mean(igd_scores), "igd_std": np.std(igd_scores),
        "hv_mean":  np.mean(hv_scores),  "hv_std":  np.std(hv_scores),
        "avg_op_probs": avg_probs,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_pareto_front(dataset_name: str,
                      all_objectives: list[np.ndarray],
                      true_pf: np.ndarray,
                      n_features: int):
    """
    Plot Pareto fronts from all runs + the union reference front.
    Mirrors Figure 1 style from the paper.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(all_objectives)))

    for i, obj in enumerate(all_objectives):
        # Sort by f2 (feature count) for a clean line
        order = np.argsort(obj[:, 1])
        ax.scatter(obj[order, 1], obj[order, 0],
                   alpha=0.6, s=30, color=colors[i],
                   label=f"Run {i+1}", zorder=3)

    # Reference / union Pareto front
    order = np.argsort(true_pf[:, 1])
    ax.plot(true_pf[order, 1], true_pf[order, 0],
            'k--', linewidth=1.5, label="Union PF", zorder=5)

    ax.set_xlabel("Number of selected features  (f2)", fontsize=12)
    ax.set_ylabel("Classification error %  (f1)", fontsize=12)
    ax.set_title(f"MOBGA-AOS Pareto Front – {dataset_name} ({n_features} features)",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = f"/mnt/user-data/outputs/pareto_{dataset_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run MOBGA-AOS feature selection")
    p.add_argument("--dataset",  default="DS02",    choices=list(DATASETS.keys()))
    p.add_argument("--data_dir", default=".",       help="Folder containing CSV files")
    p.add_argument("--runs",     default=3,  type=int)
    p.add_argument("--max_fes",  default=300_000, type=int)
    p.add_argument("--verbose",  action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        n_runs=args.runs,
        max_fes=args.max_fes,
        verbose=args.verbose,
    )