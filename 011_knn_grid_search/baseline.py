"""
Experiment 011: Optuna search for best K and buffer size for KNN-IPOP-CMA-ES.

Step 1: Run IPOP-CMA-ES baseline for all CEC 2013 functions.
Saves results to a pickle file that main.py will load for comparison.
"""

import argparse

import numpy as np
from optilab.data_classes import Bounds
from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.optimizers import IpopCmaEs
from optilab.utils import dump_to_pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run IPOP-CMA-ES baseline for all CEC 2013 functions."
    )
    parser.add_argument("dim", type=int, help="Dimensionality of benchmark functions.")
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of concurrent processes (default: 1).",
    )
    args = parser.parse_args()

    DIM = args.dim
    POPSIZE = int(4 + np.floor(3 * np.log(DIM)))
    NUM_RUNS = 51
    CALL_BUDGET = 1e4 * DIM
    TOL = 1e-8
    BOUNDS = Bounds(-100, 100)

    optimizer = IpopCmaEs(POPSIZE)
    all_results = []

    for func_num in range(1, 29):
        func = CECObjectiveFunction(2013, func_num, DIM)
        print(f"Running baseline: {func.metadata.name}")
        result = optimizer.run_optimization(
            NUM_RUNS,
            func,
            BOUNDS,
            CALL_BUDGET,
            TOL,
            num_processes=args.num_processes,
        )
        result.remove_x()
        all_results.append(result)

    dump_to_pickle(all_results, f"baseline_{DIM}d.pkl", zstd_compression=None)
    print(f"Saved baseline for {len(all_results)} functions to baseline_{DIM}d.pkl")
