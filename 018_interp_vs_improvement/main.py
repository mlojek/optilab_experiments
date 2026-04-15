"""
Experiment 018: Interpolation Rate vs KNN-IPOP-CMA-ES Improvement on CEC2013.

Hypothesis: functions where IPOP-CMA-ES generates many interpolatable points
(predictable landscape) are the same functions where KNN-IPOP-CMA-ES achieves
the largest improvement over vanilla IPOP-CMA-ES.

For each CEC2013 function:
  - Loads per-function overall interpolation % from experiment 017 (last10 pkl files)
  - Runs IpopCmaEs and KnnIpopCmaEs for num_runs independent runs each
  - Computes improvement factor = IPOP_median_best / KNN_median_best
  - Produces a scatter plot (x = interp %, y = improvement factor in log scale)
  - Saves a CSV summary
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from optilab.data_classes import Bounds
from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.optimizers.ipop_cma_es import IpopCmaEs
from optilab.optimizers.knn_ipop_cma_es import KnnIpopCmaEs
from optilab.utils import dump_to_pickle, load_from_pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../017_pdf_interpolation"))
from distribution_tracking_ipop_cma_es import GenerationRecord  # noqa: E402

FunctionResults = Tuple[float, float, float, float]  # interp_pct, ipop_median, knn_median, improvement_factor


def overall_interpolation_pct(generation_records_per_run) -> float:
    total_interpolated = sum(rec.n_interpolated for run in generation_records_per_run for rec in run)
    total_points = sum(rec.total for run in generation_records_per_run for rec in run)
    return total_interpolated / total_points * 100 if total_points > 0 else 0.0


def load_function_interpolation_pct(func_num: int, dim: int, data_dir: Path) -> float:
    pkl_path = data_dir / f"017_pdf_interp_cec2013_f{func_num:02d}_{dim}D.pkl"
    return overall_interpolation_pct(load_from_pickle(pkl_path))


def benchmark_optimizer(optimizer, func, bounds, call_budget, tolerance, num_runs, num_processes):
    return optimizer.run_optimization(
        num_runs, func, bounds, call_budget, tolerance, num_processes=num_processes
    )


def median_best_y(optimization_run) -> float:
    return float(np.median(optimization_run.bests_y(raw_values=False)))


def knn_improvement_factor(ipop_median: float, knn_median: float) -> float:
    return ipop_median / knn_median


def save_results_csv(results: Dict[str, FunctionResults], dim: int) -> str:
    csv_path = f"018_results_{dim}D.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["func_name", "interp_pct", "ipop_median", "knn_median", "improvement_factor"])
        for func_name, (interp_pct, ipop_median, knn_median, improvement) in results.items():
            writer.writerow([
                func_name,
                f"{interp_pct:.2f}",
                f"{ipop_median:.6e}",
                f"{knn_median:.6e}",
                f"{improvement:.4f}",
            ])
    return csv_path


def short_function_label(func_name: str) -> str:
    return func_name.split("_")[-1] if "_" in func_name else func_name


def save_scatter_plot(
    results: Dict[str, FunctionResults],
    dim: int,
    num_neighbors: int,
    buffer_size: int,
    buf_multiplier: int,
    num_runs: int,
) -> str:
    interpolation_pcts = [v[0] for v in results.values()]
    improvement_factors = [v[3] for v in results.values()]
    func_names = list(results.keys())

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(interpolation_pcts, improvement_factors, color="steelblue", zorder=3)
    for func_name, x, y in zip(func_names, interpolation_pcts, improvement_factors):
        ax.annotate(
            short_function_label(func_name),
            (x, y),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="No improvement (factor=1)")
    ax.set_yscale("log")
    ax.set_xlabel("Interpolation % (experiment 017, history_size=10)")
    ax.set_ylabel("Improvement factor: IPOP_median / KNN_median  (log scale)")
    ax.set_title(
        f"Interpolation rate vs KNN-IPOP improvement — CEC2013 {dim}D\n"
        f"K={num_neighbors}, buffer={buffer_size} ({buf_multiplier}×POPSIZE), {num_runs} runs"
    )
    ax.legend()
    plt.tight_layout()

    plot_path = f"018_interp_vs_improvement_{dim}D.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scatter plot of interpolation % vs KNN-IPOP improvement on CEC2013."
    )
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=51)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument(
        "--buf_multiplier",
        type=int,
        default=20,
        help="buffer_size = buf_multiplier * POPSIZE (default: 20)",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=-1,
        help="Number of KNN neighbours. -1 → DIM + 2 (default: -1)",
    )
    parser.add_argument(
        "--interp_data_dir",
        type=Path,
        default=Path(__file__).parent / "../017_pdf_interpolation/last10",
        help="Directory containing experiment 017 interpolation pkl files.",
    )
    args = parser.parse_args()

    DIM = args.dim
    BOUNDS = Bounds(-100, 100)
    POPSIZE = int(4 + np.floor(3 * np.log(DIM)))
    CALL_BUDGET = int(1e4 * DIM)
    TOL = 1e-8
    NUM_NEIGHBORS = args.k_neighbors if args.k_neighbors != -1 else DIM + 2
    BUFFER_SIZE = args.buf_multiplier * POPSIZE

    print(f"DIM={DIM}, POPSIZE={POPSIZE}, K={NUM_NEIGHBORS}, BUF={BUFFER_SIZE}")

    results: Dict[str, FunctionResults] = {}

    for func_num in range(1, 29):
        func = CECObjectiveFunction(2013, func_num, DIM)
        func_name = func.metadata.name
        print(f"\n[{func_num:02d}/28] {func_name}")

        try:
            interp_pct = load_function_interpolation_pct(func_num, DIM, args.interp_data_dir)
        except FileNotFoundError:
            print(f"  WARNING: 017 data not found for {func_name} at {DIM}D — skipping")
            continue
        print(f"  Interpolation %: {interp_pct:.1f}%")

        ipop = IpopCmaEs(population_size=POPSIZE)
        print(f"  Running IPOP-CMA-ES ({args.num_runs} runs)...")
        ipop_run = benchmark_optimizer(ipop, func, BOUNDS, CALL_BUDGET, TOL, args.num_runs, args.num_processes)
        ipop_run.remove_x()
        dump_to_pickle(ipop_run, f"018_ipop_{func_name}_{DIM}D.pkl", zstd_compression=None)

        knn = KnnIpopCmaEs(population_size=POPSIZE, num_neighbors=NUM_NEIGHBORS, buffer_size=BUFFER_SIZE)
        print(f"  Running KNN-IPOP-CMA-ES ({args.num_runs} runs)...")
        knn_run = benchmark_optimizer(knn, func, BOUNDS, CALL_BUDGET, TOL, args.num_runs, args.num_processes)
        knn_run.remove_x()
        dump_to_pickle(knn_run, f"018_knn_{func_name}_{DIM}D.pkl", zstd_compression=None)

        ipop_median = median_best_y(ipop_run)
        knn_median = median_best_y(knn_run)
        improvement = knn_improvement_factor(ipop_median, knn_median)
        print(f"  IPOP median: {ipop_median:.3e} | KNN median: {knn_median:.3e} | factor: {improvement:.3f}")

        results[func_name] = (interp_pct, ipop_median, knn_median, improvement)

    if not results:
        print("No results to plot.")
        raise SystemExit(1)

    csv_path = save_results_csv(results, DIM)
    print(f"\nSaved {csv_path}")

    plot_path = save_scatter_plot(
        results, DIM, NUM_NEIGHBORS, BUFFER_SIZE, args.buf_multiplier, args.num_runs
    )
    print(f"Saved {plot_path}")
