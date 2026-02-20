"""
main.py  –  Entry point for MOBGA-AOS experiments

Usage:
    python main.py                    # DS02, 3 runs, 300k FEs
    python main.py --dataset DS04     # single dataset
    python main.py --max_fes 50000    # smaller budget for quick testing
    python main.py --runs 5           # fewer runs
    python main.py --no_verbose       # suppress per-generation output
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all systems
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from configs.configs import DATA_PATH, DATASETS
from data_loader.data import load_dataset, normalize
from mobga_aos import MOBGA_AOS
from metrics import compute_igd, compute_hv, get_reference_point, build_true_pareto_front
from nsga_methods.crossover import OPERATOR_NAMES
from nsga_methods.evaluate import evaluate_on_test

# ── Output directories ────────────────────────────────────────────────────────
# Paths relative to this file (src/), so always correct regardless of cwd
SRC_DIR    = Path(__file__).resolve().parent
REPORT_DIR = SRC_DIR.parent / "report"
PLOTS_DIR  = REPORT_DIR / "plots"
DATA_DIR   = REPORT_DIR / "dataFrames"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Single run ────────────────────────────────────────────────────────────────

def run_once(X_train, y_train, n_features, max_fes, seed, verbose=False):
    """Run MOBGA-AOS once and return Pareto front + final operator probs."""
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
    return solutions, objectives, algo.aos.probs


# ── Experiment on one dataset ─────────────────────────────────────────────────

def run_experiment(dataset_name: str,
                   n_runs: int   = 3,
                   max_fes: int  = 300_000,
                   verbose: bool = True):
    """
    Run MOBGA-AOS n_runs times on one dataset.
    Reports IGD/HV on both training and test sets.
    Saves results to CSV and plots to PNG.
    """
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}   runs={n_runs}   maxFEs={max_fes}")
    print(f"{'='*60}")

    # ── Load data ──
    path = DATA_PATH / DATASETS[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    X_train, X_test, y_train, y_test, n_features = load_dataset(str(path))
    X_train, X_test = normalize(X_train, X_test)

    print(f"  Features: {n_features}  |  Train: {len(y_train)}  |  Test: {len(y_test)}")

    all_train_obj  = []
    all_test_obj   = []
    all_solutions  = []
    all_op_probs   = []
    all_times      = []

    for run in range(n_runs):
        t0 = time.perf_counter()
        print(f"\n  ── Run {run+1}/{n_runs} (seed={run}) ──")

        sols, train_obj, op_probs = run_once(
            X_train, y_train, n_features,
            max_fes=max_fes, seed=run, verbose=verbose
        )
        elapsed = time.perf_counter() - t0

        # ── Evaluate Pareto front solutions on TEST set ──
        # This is only done AFTER evolution — test data never seen during training
        test_obj = np.array([
            evaluate_on_test(sol, X_train, y_train, X_test, y_test)
            for sol in sols
        ])

        all_solutions.append(sols)
        all_train_obj.append(train_obj)
        all_test_obj.append(test_obj)
        all_op_probs.append(op_probs)
        all_times.append(elapsed)

        print(f"  Time: {elapsed:.1f}s  |  Pareto solutions: {len(train_obj)}")
        print(f"  Train → best_err: {train_obj[:,0].min():.3f}%  "
              f"min_feat: {train_obj[:,1].min():.0f}")
        print(f"  Test  → best_err: {test_obj[:,0].min():.3f}%  "
              f"min_feat: {test_obj[:,1].min():.0f}")

    # ── Build reference Pareto fronts (union of all runs) ──
    true_train_pf = build_true_pareto_front(all_train_obj)
    true_test_pf  = build_true_pareto_front(all_test_obj)

    # ── Compute IGD and HV ──
    train_ref = get_reference_point(all_train_obj)
    test_ref  = get_reference_point(all_test_obj)

    train_igd, train_hv = [], []
    test_igd,  test_hv  = [], []

    for t_obj, te_obj in zip(all_train_obj, all_test_obj):
        train_igd.append(compute_igd(t_obj,  true_train_pf))
        train_hv.append( compute_hv( t_obj,  train_ref))
        test_igd.append( compute_igd(te_obj, true_test_pf))
        test_hv.append(  compute_hv( te_obj, test_ref))

    # ── Print summary ──
    print(f"\n  ── Summary over {n_runs} runs ──")
    print(f"  {'Metric':<20} {'Train mean':>12} {'Train std':>10} "
          f"{'Test mean':>12} {'Test std':>10}")
    print(f"  {'-'*66}")
    print(f"  {'IGD':<20} {np.mean(train_igd):>12.4f} {np.std(train_igd):>10.4f} "
          f"{np.mean(test_igd):>12.4f} {np.std(test_igd):>10.4f}")
    print(f"  {'HV':<20} {np.mean(train_hv):>12.4f} {np.std(train_hv):>10.4f} "
          f"{np.mean(test_hv):>12.4f} {np.std(test_hv):>10.4f}")
    print(f"  {'Time (s)':<20} {np.mean(all_times):>12.1f} {np.std(all_times):>10.1f}")

    avg_probs = np.mean(all_op_probs, axis=0)
    print(f"\n  Final operator probabilities (averaged over runs):")
    for name, prob in zip(OPERATOR_NAMES, avg_probs):
        bar = "█" * int(prob * 40)
        print(f"    {name:20s} {prob:.3f}  {bar}")

    # ── Save results to CSV ──
    results = {
        "dataset":    dataset_name,
        "run":        list(range(n_runs)),
        "time_s":     all_times,
        "train_igd":  train_igd,
        "train_hv":   train_hv,
        "test_igd":   test_igd,
        "test_hv":    test_hv,
    }
    for i, name in enumerate(OPERATOR_NAMES):
        results[f"op_{name}"] = [p[i] for p in all_op_probs]

    df = pd.DataFrame(results)
    csv_path = DATA_DIR / f"results_{dataset_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved → {csv_path}")

    # ── Plot ──
    plot_pareto_front(dataset_name, all_train_obj, all_test_obj,
                      true_train_pf, n_features)

    return {
        "train_igd_mean": np.mean(train_igd), "train_igd_std": np.std(train_igd),
        "train_hv_mean":  np.mean(train_hv),  "train_hv_std":  np.std(train_hv),
        "test_igd_mean":  np.mean(test_igd),  "test_igd_std":  np.std(test_igd),
        "test_hv_mean":   np.mean(test_hv),   "test_hv_std":   np.std(test_hv),
        "avg_op_probs":   avg_probs,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_pareto_front(dataset_name: str,
                      all_train_obj: list[np.ndarray],
                      all_test_obj:  list[np.ndarray],
                      true_pf: np.ndarray,
                      n_features: int):
    """
    Two-panel plot: training Pareto fronts (left) and test Pareto fronts (right).
    Mirrors Figure 1 style from the paper.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(all_train_obj)))

    for ax, obj_list, title in zip(
        axes,
        [all_train_obj, all_test_obj],
        ["Training Set", "Test Set"]
    ):
        for i, obj in enumerate(obj_list):
            order = np.argsort(obj[:, 1])
            ax.scatter(obj[order, 1], obj[order, 0],
                       alpha=0.6, s=25, color=colors[i],
                       label=f"Run {i+1}", zorder=3)

        # Union Pareto front (only on training panel)
        if title == "Training Set":
            order = np.argsort(true_pf[:, 1])
            ax.plot(true_pf[order, 1], true_pf[order, 0],
                    'k--', linewidth=1.5, label="Union PF", zorder=5)

        ax.set_xlabel("Selected features  (f2)", fontsize=11)
        ax.set_ylabel("Classification error %  (f1)", fontsize=11)
        ax.set_title(f"{dataset_name} – {title} ({n_features} features)", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PLOTS_DIR / f"pareto_{dataset_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Run MOBGA-AOS feature selection")
    p.add_argument("--dataset",    default="DS02", choices=list(DATASETS.keys()))
    p.add_argument("--runs",       default=3,      type=int)
    p.add_argument("--max_fes",    default=300_000, type=int)
    p.add_argument("--no_verbose", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        dataset_name=args.dataset,
        n_runs=args.runs,
        max_fes=args.max_fes,
        verbose=not args.no_verbose,
    )