"""
Experiment 018: Median value vs interpolation % for IPOP and KNN-IPOP-CMA-ES.

Two point series per plot:
  - IPOP-CMA-ES medians (orange)
  - KNN-IPOP-CMA-ES medians at 20× buffer (blue)

X axis: interpolation % from experiment 017
Y axis: best-y median (log scale)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optilab.functions.benchmarks import CECObjectiveFunction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../017_pdf_interpolation"))
from distribution_tracking_ipop_cma_es import GenerationRecord  # noqa: E402


def overall_interpolation_pct(generation_records_per_run: list) -> float:
    from optilab.utils import load_from_pickle
    total_interpolated = sum(rec.n_interpolated for run in generation_records_per_run for rec in run)
    total_points = sum(rec.total for run in generation_records_per_run for rec in run)
    return total_interpolated / total_points * 100 if total_points > 0 else 0.0


def load_interp_pct(func_num: int, dim: int, data_dir: Path) -> float:
    from optilab.utils import load_from_pickle
    pkl_path = data_dir / f"017_pdf_interp_cec2013_f{func_num:02d}_{dim}D.pkl"
    return overall_interpolation_pct(load_from_pickle(pkl_path))


def load_medians_from_stats_csv(
    stats_csv: Path, ipop_col: str, knn_col: str
) -> dict[str, tuple[float, float]]:
    medians: dict[str, tuple[float, float]] = {}
    with stats_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["stat"] != "y_median":
                continue
            func_name = row["function"]
            if ipop_col not in row:
                raise KeyError(f"Column '{ipop_col}' not found. Available: {list(row.keys())}")
            if knn_col not in row:
                raise KeyError(f"Column '{knn_col}' not found. Available: {list(row.keys())}")
            medians[func_name] = (float(row[ipop_col]), float(row[knn_col]))
    return medians


def load_medians_from_stats_dir(
    stats_dir: Path, ipop_col: str, knn_col: str
) -> dict[str, tuple[float, float]]:
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


def short_label(func_name: str) -> str:
    return func_name.split("_")[-1] if "_" in func_name else func_name


def save_median_plot(
    data: list[tuple[str, float, float, float]],  # (func_name, interp_pct, ipop_med, knn_med)
    dim: int,
    num_neighbors: int,
    buffer_size: int,
    buf_multiplier: int,
) -> Path:
    interp_pcts = [d[1] for d in data]
    ipop_meds = [d[2] for d in data]
    knn_meds = [d[3] for d in data]
    func_names = [d[0] for d in data]

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(interp_pcts, ipop_meds, color="darkorange", label="IPOP-CMA-ES", zorder=3)
    ax.scatter(interp_pcts, knn_meds, color="steelblue", label=f"KNN-IPOP (k={num_neighbors}, buf={buf_multiplier}×)", zorder=3)

    for name, x, yi, yk in zip(func_names, interp_pcts, ipop_meds, knn_meds):
        label = short_label(name)
        ax.annotate(label, (x, yi), textcoords="offset points", xytext=(4, 3), fontsize=6, color="darkorange")
        ax.annotate(label, (x, yk), textcoords="offset points", xytext=(4, 3), fontsize=6, color="steelblue")

    ax.set_yscale("log")
    ax.set_xlabel("Interpolation % (experiment 017, history_size=10)")
    ax.set_ylabel("Best-y median (log scale)")
    ax.set_title(
        f"Median best-y vs interpolation rate — CEC2013 {dim}D\n"
        f"K={num_neighbors}, buffer={buffer_size} ({buf_multiplier}×POPSIZE)"
    )
    ax.legend()
    plt.tight_layout()

    plot_path = Path(f"018_medians_{dim}D.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot IPOP and KNN-IPOP medians vs interpolation % for CEC2013."
    )
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--start_from", type=int, default=1)
    parser.add_argument("--stop_at", type=int, default=28)
    parser.add_argument("--buf_multiplier", type=int, default=20)
    parser.add_argument("--k_neighbors", type=int, default=-1, help="-1 → DIM + 2")
    parser.add_argument(
        "--interp_data_dir",
        type=Path,
        default=Path(__file__).parent / "../017_pdf_interpolation/last10",
    )
    parser.add_argument("--stats_csv", type=Path, default=None,
                        help="Aggregated stats CSV (wide format).")
    parser.add_argument("--stats_dir", type=Path, default=None,
                        help="Directory of per-function *.stats.csv files (long format).")
    parser.add_argument("--remove_outliers", action="store_true",
                        help="Drop functions where either median is ≤ 0 (unplottable on log scale).")
    args = parser.parse_args()

    if args.stats_csv and args.stats_dir:
        print("ERROR: specify only one of --stats_csv or --stats_dir")
        raise SystemExit(1)
    if not args.stats_csv and not args.stats_dir:
        print("ERROR: one of --stats_csv or --stats_dir is required")
        raise SystemExit(1)

    DIM = args.dim
    POPSIZE = int(4 + np.floor(3 * np.log(DIM)))
    NUM_NEIGHBORS = args.k_neighbors if args.k_neighbors != -1 else DIM + 2
    BUFFER_SIZE = args.buf_multiplier * POPSIZE
    IPOP_COL = "ipop-cma-es"
    KNN_COL = f"knn{NUM_NEIGHBORS}b{BUFFER_SIZE}-ipop-cma-es"

    print(f"DIM={DIM}, POPSIZE={POPSIZE}, K={NUM_NEIGHBORS}, BUF={BUFFER_SIZE}")
    print(f"IPOP column: {IPOP_COL}  |  KNN column: {KNN_COL}")

    if args.stats_csv:
        print(f"Loading medians from {args.stats_csv}")
        medians = load_medians_from_stats_csv(args.stats_csv, IPOP_COL, KNN_COL)
    else:
        print(f"Loading medians from {args.stats_dir}")
        medians = load_medians_from_stats_dir(args.stats_dir, IPOP_COL, KNN_COL)
    print(f"  Loaded {len(medians)} functions")

    data: list[tuple[str, float, float, float]] = []

    for func_num in range(args.start_from, args.stop_at + 1):
        func = CECObjectiveFunction(2013, func_num, DIM)
        func_name = func.metadata.name

        if func_name not in medians:
            print(f"  WARNING: {func_name} not in stats — skipping")
            continue

        try:
            from optilab.utils import load_from_pickle
            interp_pct = load_interp_pct(func_num, DIM, args.interp_data_dir)
        except FileNotFoundError:
            print(f"  WARNING: 017 data not found for {func_name} — skipping")
            continue

        ipop_med, knn_med = medians[func_name]

        if args.remove_outliers and (ipop_med <= 0 or knn_med <= 0):
            print(f"  Skipping {func_name}: non-positive median (unplottable on log scale)")
            continue

        data.append((func_name, interp_pct, ipop_med, knn_med))

    if not data:
        print("No data to plot.")
        raise SystemExit(1)

    print(f"\nPlotting {len(data)} functions...")
    plot_path = save_median_plot(data, DIM, NUM_NEIGHBORS, BUFFER_SIZE, args.buf_multiplier)
    print(f"Saved {plot_path}")
