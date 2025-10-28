"""
Comparing IPOP-CMA-ES with LMM-IPOP-CMA-ES on CEC2013/CEC2017 benchmark.
"""

import argparse

import numpy as np
from optilab.data_classes import Bounds
from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.optimizers import IpopCmaEs, LmmIpopCmaEs, LmmCmaEs
from optilab.utils import dump_to_pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "year",
        type=int,
        choices=[2013, 2017],
        help="Year of CEC benchmark, either 2013 or 2017.",
    )
    parser.add_argument(
        "dim",
        type=int,
        help="Dimensionality of benchmark functions.",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=1,
        help="Function number to start from.",
    )
    parser.add_argument(
        "--stop_at",
        type=int,
        default=100,
        help="Function number to stop at.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of concurrent processes to use.",
    )
    args = parser.parse_args()

    # optimized problem
    DIM = args.dim
    BOUNDS = Bounds(-100, 100)
    FUNCS = {
        2013: [
            CECObjectiveFunction(2013, n, DIM)
            for n in range(args.start_from, min(args.stop_at + 1, 29))
        ],
        2017: [
            CECObjectiveFunction(2017, n, DIM)
            for n in range(args.start_from, min(args.stop_at + 1, 30))
        ],
    }
    TARGET = 0.0

    # hyperparams:
    POPSIZE = int(4 + np.floor(3 * np.log(DIM)))
    NUM_NEIGHBORS = DIM + 2
    POLYNOMIAL_DIM = 2
    NUM_RUNS = 51
    CALL_BUDGET = 1e4 * DIM
    TOL = 1e-8

    for func in FUNCS[args.year]:
        print(func.metadata.name)
        results = []

        cmaes_optimizer = IpopCmaEs(POPSIZE)
        print(cmaes_optimizer.metadata.name)
        cmaes_results = cmaes_optimizer.run_optimization(
            NUM_RUNS, func, BOUNDS, CALL_BUDGET, TOL, num_processes=args.num_processes
        )
        cmaes_results.remove_x()
        results.append(cmaes_results)

        lmm_optimizer = LmmCmaEs(POPSIZE, 10, POLYNOMIAL_DIM)
        print(lmm_optimizer.metadata.name)
        lmm_results = lmm_optimizer.run_optimization(
            NUM_RUNS, func, BOUNDS, CALL_BUDGET, TOL, num_processes=args.num_processes
        )
        lmm_results.remove_x()
        results.append(lmm_results)

        dump_to_pickle(results, f"008_lmm_vs_ipop_{func.metadata.name}_{DIM}.pkl", zstd_compression=None)
