"""
Experiment 009 — interpolation benchmark.

For each (m, sigma, C) state: sample 2*N points from N(m, sigma^2*C), randomly
split into equal train/test halves, and measure each surrogate's accuracy.
"""

import argparse
from functools import partial
from multiprocessing.pool import Pool
from typing import Dict, List

import numpy as np
import pandas as pd

from optilab.functions.benchmarks import CECObjectiveFunction

from benchmark_utils import (
    BOUNDS,
    create_surrogates,
    default_pop_size,
    evaluate_surrogate,
    load_dataset,
    sample_population,
    split_population,
)


def evaluate_record(record: Dict, pop_size: int) -> List[Dict]:
    function_num = record["function_num"]
    dim = record["dim"]
    m, sigma, C = record["m"], record["sigma"], record["C"]
    n = pop_size or default_pop_size(dim)

    func = CECObjectiveFunction(2013, function_num, dim)
    cov = (sigma**2) * C

    try:
        all_points = sample_population(m, cov, 2 * n, BOUNDS, func)
        train_set, test_set = split_population(all_points, m, cov, extrapolation=False)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"  WARNING: f{function_num:02d} dim={dim} — sampling failed: {e}")
        return []

    results = []
    for surrogate_name, surrogate in create_surrogates(dim, C):
        try:
            mape, spearman = evaluate_surrogate(surrogate, train_set, test_set)
            results.append(
                {
                    "function_num": function_num,
                    "dim": dim,
                    "surrogate": surrogate_name,
                    "mape": mape,
                    "spearman": spearman,
                }
            )
        except Exception as e:
            print(f"  WARNING: f{function_num:02d} dim={dim} {surrogate_name} — {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark surrogate interpolation.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--pop_size",
        type=int,
        default=0,
        help="N for train/test sets (default: dim*(dim+3)//2+2, i.e. LWR neighbor count).",
    )
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--output", type=str, default="interpolation_results.csv")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} records from {args.dataset}")

    worker = partial(evaluate_record, pop_size=args.pop_size)
    all_results: List[Dict] = []

    if args.num_processes > 1:
        with Pool(processes=args.num_processes) as pool:
            for batch in pool.map(worker, dataset):
                all_results.extend(batch)
    else:
        for i, rec in enumerate(dataset):
            all_results.extend(worker(rec))
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} records...")

    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} results to {args.output}")

    if not df.empty:
        print("\n=== Interpolation — average of per-function medians ===")
        per_fn = df.groupby(["surrogate", "function_num"])[["mape", "spearman"]].median()
        summary = per_fn.groupby("surrogate").mean()
        print(summary.to_string())
