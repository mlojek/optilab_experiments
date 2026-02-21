"""
Experiment 009, Script 3: Test surrogate extrapolation accuracy.

For each (m, sigma, C) record in the dataset:
  - Training set is sampled from N(m, sigma^2 * C) with Mahalanobis distance <= 1
    (within one sigma of the mean)
  - Test set is sampled from N(m, sigma^2 * C) with Mahalanobis distance > 1
    (beyond one sigma, testing out-of-distribution generalization)

Evaluates each surrogate's prediction accuracy on the test set using MAPE
and Spearman rank correlation.
"""

import argparse
from functools import partial
from multiprocessing.pool import Pool
from typing import Dict, List

import numpy as np
import pandas as pd

from optilab.functions.benchmarks import CECObjectiveFunction

from _common import (
    BOUNDS,
    create_surrogates,
    evaluate_surrogate,
    load_dataset,
    sample_population,
)


def evaluate_record(
    record: Dict,
    pop_size: int,
) -> List[Dict]:
    """
    Evaluate all surrogates on one dataset record (extrapolation).

    Training within 1 sigma, testing beyond 1 sigma (same covariance).

    Args:
        record: Dict with function_num, dim, m, sigma, C.
        pop_size: Population size for train and test sets.

    Returns:
        List of result dicts with metrics for each surrogate.
    """
    function_num = record["function_num"]
    dim = record["dim"]
    m = record["m"]
    sigma = record["sigma"]
    C = record["C"]

    func = CECObjectiveFunction(2013, function_num, dim)
    cov = (sigma**2) * C

    try:
        train_set = sample_population(
            m, cov, pop_size, BOUNDS, func, max_mahalanobis=1.0
        )
        test_set = sample_population(
            m, cov, pop_size, BOUNDS, func, min_mahalanobis=1.0
        )
    except (np.linalg.LinAlgError, ValueError) as e:
        print(
            f"  WARNING: f{function_num:02d} dim={dim} sigma={sigma:.4f} "
            f"— sampling failed: {e}"
        )
        return []

    surrogates = create_surrogates(dim, C)
    results = []

    for surrogate_name, surrogate in surrogates:
        try:
            mape, spearman = evaluate_surrogate(surrogate, train_set, test_set)
            results.append(
                {
                    "function_num": function_num,
                    "dim": dim,
                    "sigma": sigma,
                    "surrogate": surrogate_name,
                    "mape": mape,
                    "spearman": spearman,
                }
            )
        except Exception as e:
            print(
                f"  WARNING: f{function_num:02d} dim={dim} {surrogate_name} "
                f"— evaluation failed: {e}"
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test surrogate function extrapolation accuracy."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSON from create_dataset.py.",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=50,
        help="Population size for train/test sets (default: 50).",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="extrapolation_results.csv",
        help="Output CSV path (default: extrapolation_results.csv).",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    print(
        f"Loaded {len(dataset)} records from {args.dataset}, "
        f"pop_size={args.pop_size}"
    )

    worker = partial(evaluate_record, pop_size=args.pop_size)
    all_results: List[Dict] = []

    if args.num_processes > 1:
        with Pool(processes=args.num_processes) as pool:
            batches = pool.map(worker, dataset)
        for batch in batches:
            all_results.extend(batch)
    else:
        for i, rec in enumerate(dataset):
            batch = worker(rec)
            all_results.extend(batch)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} records...")

    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} results to {args.output}")

    # Print summary
    if not df.empty:
        print("\n=== Extrapolation Summary (mean across all records) ===")
        summary = df.groupby("surrogate")[["mape", "spearman"]].mean()
        print(summary.to_string())
