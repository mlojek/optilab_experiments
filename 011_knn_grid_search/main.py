"""
Experiment 011: Optuna search for best K and buffer size for KNN-IPOP-CMA-ES.

Uses Optuna to efficiently explore the (K, bufsize) hyperparameter space.
For each trial, runs KNN-IPOP-CMA-ES on all 28 CEC 2013 functions (51 runs each),
compares with the IPOP-CMA-ES baseline using the Mann-Whitney U test, and
maximizes the number of functions where KNN-IPOP significantly improves.

Prerequisites: run baseline.py first to generate the baseline results.

Usage:
    python baseline.py 10 --num_processes 51
    python main.py 10 --num_processes 51 --n_trials 200
"""

import argparse
import os

import numpy as np
import optuna
from scipy.stats import mannwhitneyu

from optilab.data_classes import Bounds
from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.optimizers import KnnIpopCmaEs
from optilab.utils import load_from_pickle


def objective(
    trial: optuna.Trial,
    baseline_results: list,
    dim: int,
    popsize: int,
    num_processes: int,
) -> int:
    """
    Optuna objective: for a given (K, bufsize), run KNN-IPOP-CMA-ES on all 28
    CEC 2013 functions and count the number of significant improvements over
    IPOP-CMA-ES baseline (Mann-Whitney U test, alpha=0.05).

    Args:
        trial: Optuna trial object.
        baseline_results: List of 28 OptimizationRun objects (one per function).
        dim: Dimensionality.
        popsize: Population size.
        num_processes: Number of parallel processes for optimization runs.

    Returns:
        Number of functions where KNN-IPOP significantly improves over IPOP.
    """
    k = trial.suggest_int("k", dim, 100 * dim)
    buf_multiplier = trial.suggest_int("buf_multiplier", 1, 200)
    bufsize = buf_multiplier * popsize

    bounds = Bounds(-100, 100)
    call_budget = 1e4 * dim
    tol = 1e-8
    num_runs = 51

    optimizer = KnnIpopCmaEs(popsize, k, bufsize)
    num_improvements = 0

    for func_num in range(1, 29):
        func = CECObjectiveFunction(2013, func_num, dim)
        baseline = baseline_results[func_num - 1]

        try:
            knn_result = optimizer.run_optimization(
                num_runs,
                func,
                bounds,
                call_budget,
                tol,
                num_processes=num_processes,
            )
        except Exception as e:
            print(f"  f{func_num:02d} failed: {e}")
            continue

        baseline_y = baseline.bests_y()
        knn_y = knn_result.bests_y()

        # One-sided Mann-Whitney U test: is KNN-IPOP better (lower y)?
        _, p_value = mannwhitneyu(knn_y, baseline_y, alternative="less")

        if p_value < 0.05:
            num_improvements += 1

    print(
        f"Trial {trial.number}: k={k}, buf={buf_multiplier}*pop={bufsize} "
        f"-> {num_improvements}/28 improvements"
    )
    return num_improvements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna search for best K and buffer size for KNN-IPOP-CMA-ES."
    )
    parser.add_argument("dim", type=int, help="Dimensionality of benchmark functions.")
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100).",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of concurrent processes for CMA-ES runs (default: 1).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline pickle (default: baseline_{dim}d.pkl).",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Optuna study name (default: knn_grid_search_{dim}d).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL for distributed runs (e.g. sqlite:///study.db).",
    )
    args = parser.parse_args()

    DIM = args.dim
    POPSIZE = int(4 + np.floor(3 * np.log(DIM)))

    baseline_path = args.baseline or f"baseline_{DIM}d.pkl"
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Baseline file '{baseline_path}' not found. "
            f"Run baseline.py first: python baseline.py {DIM} --num_processes N"
        )

    print(f"Loading baseline from {baseline_path}...")
    baseline_results = load_from_pickle(baseline_path)
    print(f"Loaded {len(baseline_results)} baseline results.")

    study_name = args.study_name or f"knn_grid_search_{DIM}d"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial, baseline_results, DIM, POPSIZE, args.num_processes
        ),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 60)
    print("Best trial:")
    best = study.best_trial
    print(f"  k = {best.params['k']}")
    print(f"  buf_multiplier = {best.params['buf_multiplier']}")
    print(f"  bufsize = {best.params['buf_multiplier'] * POPSIZE}")
    print(f"  improvements = {best.value}/28")
    print("=" * 60)

    # Save top 10 results to CSV
    df = study.trials_dataframe()
    df = df.sort_values("value", ascending=False)
    output_path = f"optuna_results_{DIM}d.csv"
    df.to_csv(output_path, index=False)
    print(f"\nAll trial results saved to {output_path}")
