"""
Experiment 009 — final summary.

Reads interpolation and extrapolation result CSVs and prints the key metric:
average of per-function medians of MAPE and Spearman, per surrogate per dim.

Usage:
    python main.py --interp interpolation_10d.csv interpolation_30d.csv \
                   --extrap extrapolation_10d.csv  extrapolation_30d.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def summarize(csv_path: str, mode: str) -> None:
    df = pd.read_csv(csv_path)
    dim = df["dim"].iloc[0]
    per_fn = df.groupby(["surrogate", "function_num"])[["mape", "spearman"]].median()
    summary = per_fn.groupby("surrogate").mean()
    print(f"\n=== {mode.capitalize()} | dim={dim} ===")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print mmbench final summary.")
    parser.add_argument("--interp", nargs="+", default=[], metavar="CSV")
    parser.add_argument("--extrap", nargs="+", default=[], metavar="CSV")
    args = parser.parse_args()

    for path in args.interp:
        summarize(path, "interpolation")
    for path in args.extrap:
        summarize(path, "extrapolation")
