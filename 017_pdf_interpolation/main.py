"""
Experiment 017: PDF Interpolation Analysis of IPOP-CMA-ES on CEC2013.

For each CEC2013 function, runs IPOP-CMA-ES N times and measures what fraction
of newly generated points in each generation could have been interpolated from
the distributions of all previous generations.

A point is considered interpolatable if the PDF of any previous generation's
multivariate Gaussian is >= pdf_threshold at that point.
"""

import argparse
from multiprocessing import Pool
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from optilab.data_classes import Bounds
from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.utils import dump_to_pickle

from distribution_tracking_ipop_cma_es import GenerationRecord, run_distribution_tracking_ipop


def _run_single(args):
    """Worker function for multiprocessing."""
    function, bounds, call_budget, tolerance, population_size, pdf_threshold, history_size = args
    return run_distribution_tracking_ipop(
        function,
        bounds,
        call_budget,
        tolerance,
        population_size=population_size,
        pdf_threshold=pdf_threshold,
        history_size=history_size,
    )


def plot_function_results(
    runs: List[List[GenerationRecord]],
    func_name: str,
    dim: int,
    output_path: str,
) -> None:
    """
    Generate a two-line plot showing per-generation and cumulative % interpolated.

    Args:
        runs: List of runs, each being a list of GenerationRecord (one per generation).
        func_name: Function name for the title.
        dim: Dimensionality for the title.
        output_path: File path to save the plot.
    """
    max_gens = max(len(run) for run in runs)

    # Per-generation %: average over runs that have reached generation g.
    per_gen_pct = []
    for g in range(max_gens):
        pcts = [
            run[g].pct_interpolated
            for run in runs
            if g < len(run)
        ]
        per_gen_pct.append(np.mean(pcts))

    # Cumulative %: compute per-run running average, then average across runs.
    run_cumulative_pcts = []
    for run in runs:
        total_interp = 0
        total_pts = 0
        cum = []
        for rec in run:
            total_interp += rec.n_interpolated
            total_pts += rec.total
            cum.append(total_interp / total_pts * 100 if total_pts > 0 else 0.0)
        run_cumulative_pcts.append(cum)

    cumulative_pct = []
    for g in range(max_gens):
        vals = [run[g] for run in run_cumulative_pcts if g < len(run)]
        cumulative_pct.append(np.mean(vals))

    # Mean sigma per generation, averaged across runs.
    mean_sigma = []
    for g in range(max_gens):
        vals = [run[g].sigma for run in runs if g < len(run)]
        mean_sigma.append(np.mean(vals))

    # Overall % across all runs.
    total_interp = sum(rec.n_interpolated for run in runs for rec in run)
    total_pts = sum(rec.total for run in runs for rec in run)
    overall_pct = total_interp / total_pts * 100 if total_pts > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 5))
    gens = range(max_gens)
    ax.plot(gens, per_gen_pct, color="blue", label="Per-generation % interpolated")
    ax.plot(gens, cumulative_pct, color="orange", label="Cumulative % interpolated")
    ax.set_xlabel("Generation")
    ax.set_ylabel("% interpolated")
    ax.set_ylim(0, 100)

    ax2 = ax.twinx()
    ax2.plot(gens, mean_sigma, color="red", linewidth=0.8, label="sigma")
    ax2.set_yscale("log")
    ax2.set_ylabel("sigma (log₁₀ scale)", color="red")
    ax2.tick_params(axis="y", colors="red")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    ax.set_title(f"{func_name} {dim}D — {overall_pct:.1f}% of all points interpolated")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF interpolation analysis of IPOP-CMA-ES on CEC2013."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=10,
        help="Dimensionality of benchmark functions (default: 10).",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=51,
        help="Number of independent runs per function (default: 51).",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--pdf_threshold",
        type=float,
        default=0.005,
        help="PDF threshold for classifying a point as interpolatable (default: 0.005).",
    )
    parser.add_argument(
        "--history_size",
        type=int,
        default=10,
        help="Number of most recent populations to compare against (default: 10). Use -1 for all.",
    )
    args = parser.parse_args()

    DIM = args.dim
    BOUNDS = Bounds(-100, 100)
    POPSIZE = int(4 + np.floor(3 * np.log(DIM)))
    CALL_BUDGET = int(1e4 * DIM)
    TOL = 1e-8

    functions = [CECObjectiveFunction(2013, n, DIM) for n in [16, 27, 28]]

    for func in functions:
        print(f"\n{func.metadata.name}")

        worker_args = [
            (func, BOUNDS, CALL_BUDGET, TOL, POPSIZE, args.pdf_threshold, args.history_size)
            for _ in range(args.num_runs)
        ]

        with Pool(args.num_processes) as pool:
            runs = list(
                tqdm(
                    pool.imap(_run_single, worker_args),
                    total=args.num_runs,
                    desc="  runs",
                    unit="run",
                )
            )

        base_name = f"017_pdf_interp_{func.metadata.name}_{DIM}D"
        dump_to_pickle(runs, f"{base_name}.pkl", zstd_compression=None)

        plot_function_results(runs, func.metadata.name, DIM, f"{base_name}.png")
        print(f"  Saved {base_name}.pkl and {base_name}.png")
