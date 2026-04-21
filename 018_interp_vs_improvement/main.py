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
import gc
import os
import sys
from pathlib import Path

# Must be set before numpy/BLAS are imported in worker processes.
# Without this, each worker spawns 10 BLAS threads → N_processes × 10 threads
# compete for 10 cores → scheduler thrashes → appears to hang.
def _apply_blas_thread_limit(num_processes: int) -> None:
    if num_processes > 1:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = "1"

import matplotlib.pyplot as plt
import numpy as np

from optilab.data_classes import Bounds
from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.optimizers.ipop_cma_es import IpopCmaEs
from optilab.optimizers.knn_ipop_cma_es import KnnIpopCmaEs
from optilab.utils import dump_to_pickle, load_from_pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../017_pdf_interpolation"))
from distribution_tracking_ipop_cma_es import GenerationRecord  # noqa: E402

# interp_pct, ipop_median, knn_median, improvement_factor
FunctionResults = tuple[float, float, float, float]


def overall_interpolation_pct(generation_records_per_run: list) -> float:
    total_interpolated = sum(rec.n_interpolated for run in generation_records_per_run for rec in run)
    total_points = sum(rec.total for run in generation_records_per_run for rec in run)
    return total_interpolated / total_points * 100 if total_points > 0 else 0.0


def load_function_interpolation_pct(func_num: int, dim: int, data_dir: Path) -> float:
    pkl_path = data_dir / f"017_pdf_interp_cec2013_f{func_num:02d}_{dim}D.pkl"
    return overall_interpolation_pct(load_from_pickle(pkl_path))


def median_best_y(optimization_run) -> float:
    return float(np.median(optimization_run.bests_y(raw_values=False)))


def run_and_get_median(optimizer, func, bounds, call_budget, tolerance, num_runs, num_processes, pkl_path: Path) -> float:
    """Run optimizer, save pkl, extract median, then free memory immediately."""
    run = optimizer.run_optimization(num_runs, func, bounds, call_budget, tolerance, num_processes=num_processes)
    run.remove_x()
    dump_to_pickle(run, str(pkl_path), zstd_compression=None)
    median = median_best_y(run)
    del run
    gc.collect()
    return median


def load_median_from_pkl(pkl_path: Path) -> float:
    run = load_from_pickle(pkl_path)
    median = median_best_y(run)
    del run
    gc.collect()
    return median


def load_medians_from_stats_csv(
    stats_csv: Path, ipop_col: str, knn_col: str
) -> dict[str, tuple[float, float]]:
    """Return {func_name: (ipop_median, knn_median)} from an aggregated_stats.csv
    (wide format: columns are optimizer names, rows are functions × stats)."""
    medians: dict[str, tuple[float, float]] = {}
    with stats_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["stat"] != "y_median":
                continue
            func_name = row["function"]
            if ipop_col not in row:
                raise KeyError(f"Column '{ipop_col}' not found in {stats_csv}")
            if knn_col not in row:
                raise KeyError(f"Column '{knn_col}' not found in {stats_csv}. Available: {list(row.keys())}")
            medians[func_name] = (float(row[ipop_col]), float(row[knn_col]))
    return medians


def load_medians_from_stats_dir(
    stats_dir: Path, ipop_col: str, knn_col: str
) -> dict[str, tuple[float, float]]:
    """Return {func_name: (ipop_median, knn_median)} from a directory of per-function
    stats CSVs (long format: each row is one optimizer, columns include model/y_median)."""
    medians: dict[str, tuple[float, float]] = {}
    for path in sorted(stats_dir.glob("*.stats.csv")):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = {row["model"]: row for row in reader}
        if ipop_col not in rows:
            raise KeyError(f"Model '{ipop_col}' not found in {path}")
        if knn_col not in rows:
            raise KeyError(f"Model '{knn_col}' not found in {path}. Available: {list(rows.keys())}")
        func_name = rows[ipop_col]["function"]
        medians[func_name] = (float(rows[ipop_col]["y_median"]), float(rows[knn_col]["y_median"]))
    return medians


def knn_improvement_factor(ipop_median: float, knn_median: float) -> float:
    return ipop_median / knn_median


def save_results_csv(results: dict[str, FunctionResults], dim: int) -> Path:
    csv_path = Path(f"018_results_{dim}D.csv")
    with csv_path.open("w", newline="") as f:
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
    results: dict[str, FunctionResults],
    dim: int,
    num_neighbors: int,
    buffer_size: int,
    buf_multiplier: int,
    num_runs: int,
) -> Path:
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

    plot_path = Path(f"018_interp_vs_improvement_{dim}D.png")
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
    parser.add_argument("--start_from", type=int, default=1, help="First CEC function number to run (default: 1)")
    parser.add_argument("--stop_at", type=int, default=28, help="Last CEC function number to run (default: 28)")
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
    parser.add_argument(
        "--stats_csv",
        type=Path,
        default=None,
        help="Path to aggregated_stats.csv (wide format). "
             "When provided, medians are read directly instead of running optimizers.",
    )
    parser.add_argument(
        "--stats_dir",
        type=Path,
        default=None,
        help="Path to a directory of per-function *.stats.csv files (long format). "
             "Alternative to --stats_csv.",
    )
    parser.add_argument(
        "--remove_outliers",
        action="store_true",
        help="Drop functions whose improvement factor is outside [0.1, 10] (one order of magnitude from 1).",
    )
    args = parser.parse_args()
    _apply_blas_thread_limit(args.num_processes)

    DIM = args.dim
    BOUNDS = Bounds(-100, 100)
    POPSIZE = int(4 + np.floor(3 * np.log(DIM)))
    CALL_BUDGET = int(1e4 * DIM)
    TOL = 1e-8
    NUM_NEIGHBORS = args.k_neighbors if args.k_neighbors != -1 else DIM + 2
    BUFFER_SIZE = args.buf_multiplier * POPSIZE

    IPOP_COL = "ipop-cma-es"
    KNN_COL = f"knn{NUM_NEIGHBORS}b{BUFFER_SIZE}-ipop-cma-es"
    print(f"DIM={DIM}, POPSIZE={POPSIZE}, K={NUM_NEIGHBORS}, BUF={BUFFER_SIZE}")
    print(f"IPOP column: {IPOP_COL}  |  KNN column: {KNN_COL}")

    if args.stats_csv and args.stats_dir:
        print("ERROR: specify only one of --stats_csv or --stats_dir")
        raise SystemExit(1)

    preloaded_medians: dict[str, tuple[float, float]] | None = None
    if args.stats_csv is not None:
        print(f"Loading medians from {args.stats_csv}")
        preloaded_medians = load_medians_from_stats_csv(args.stats_csv, IPOP_COL, KNN_COL)
        print(f"  Loaded {len(preloaded_medians)} functions")
    elif args.stats_dir is not None:
        print(f"Loading medians from {args.stats_dir}")
        preloaded_medians = load_medians_from_stats_dir(args.stats_dir, IPOP_COL, KNN_COL)
        print(f"  Loaded {len(preloaded_medians)} functions")

    results: dict[str, FunctionResults] = {}

    for func_num in range(args.start_from, args.stop_at + 1):
        func = CECObjectiveFunction(2013, func_num, DIM)
        func_name = func.metadata.name
        print(f"\n[{func_num:02d}/28] {func_name}")

        try:
            interp_pct = load_function_interpolation_pct(func_num, DIM, args.interp_data_dir)
        except FileNotFoundError:
            print(f"  WARNING: 017 data not found for {func_name} at {DIM}D — skipping")
            continue
        print(f"  Interpolation %: {interp_pct:.1f}%")

        if preloaded_medians is not None:
            if func_name not in preloaded_medians:
                print(f"  WARNING: {func_name} not in stats CSV — skipping")
                continue
            ipop_median, knn_median = preloaded_medians[func_name]
            print(f"  (from CSV) IPOP median: {ipop_median:.3e} | KNN median: {knn_median:.3e}")
        else:
            ipop_pkl = Path(f"018_ipop_{func_name}_{DIM}D.pkl")
            if ipop_pkl.exists():
                print(f"  Loading IPOP-CMA-ES results from {ipop_pkl}")
                ipop_median = load_median_from_pkl(ipop_pkl)
            else:
                ipop = IpopCmaEs(population_size=POPSIZE)
                print(f"  Running IPOP-CMA-ES ({args.num_runs} runs)...")
                ipop_median = run_and_get_median(ipop, func, BOUNDS, CALL_BUDGET, TOL, args.num_runs, args.num_processes, ipop_pkl)

            knn_pkl = Path(f"018_knn_{func_name}_{DIM}D.pkl")
            if knn_pkl.exists():
                print(f"  Loading KNN-IPOP-CMA-ES results from {knn_pkl}")
                knn_median = load_median_from_pkl(knn_pkl)
            else:
                knn = KnnIpopCmaEs(population_size=POPSIZE, num_neighbors=NUM_NEIGHBORS, buffer_size=BUFFER_SIZE)
                print(f"  Running KNN-IPOP-CMA-ES ({args.num_runs} runs)...")
                knn_median = run_and_get_median(knn, func, BOUNDS, CALL_BUDGET, TOL, args.num_runs, args.num_processes, knn_pkl)

        improvement = knn_improvement_factor(ipop_median, knn_median)
        print(f"  factor: {improvement:.3f}")

        results[func_name] = (interp_pct, ipop_median, knn_median, improvement)

    if not results:
        print("No results to plot.")
        raise SystemExit(1)

    if args.remove_outliers:
        before = len(results)
        results = {
            name: vals for name, vals in results.items()
            if 0.1 <= vals[3] <= 10.0
        }
        print(f"Outlier removal: {before - len(results)} dropped, {len(results)} remaining")

    csv_path = save_results_csv(results, DIM)
    print(f"\nSaved {csv_path}")

    plot_path = save_scatter_plot(
        results, DIM, NUM_NEIGHBORS, BUFFER_SIZE, args.buf_multiplier, args.num_runs
    )
    print(f"Saved {plot_path}")
