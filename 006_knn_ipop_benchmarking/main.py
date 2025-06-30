"""
Benchmarking of CMA-ES with KNN metamodel on CEC2013/CEC2017 benchmark.
"""

import argparse

import numpy as np
from optilab.data_classes import Bounds
from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.optimizers import CmaEs, IpopCmaEs, KnnCmaEs, LmmCmaEs
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
    parser.add_argument(
        "--with_lmm",
        action="store_true",
        help="If specified, LMM-CMA-ES will also be ran.",
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
    POPSIZE = 4 + np.floor(3 * np.log(DIM))
    NUM_NEIGHBORS = DIM + 2
    BUFFER_SIZES = [m * POPSIZE for m in [2, 5, 10, 20, 30, 50]]
    NUM_RUNS = 51
    CALL_BUDGET = 1e4 * DIM
    TOL = 1e-8
    SIGMA0 = len(BOUNDS) / 2

    for func in FUNCS[args.year]:
        print(func.metadata.name)
        results = []

        ipop_optimizer = IpopCmaEs(POPSIZE)
        print(ipop_optimizer.metadata.name)
        ipop_results = ipop_optimizer.run_optimization(
            NUM_RUNS, func, BOUNDS, CALL_BUDGET, TOL, num_processes=args.num_processes
        )
        results.append(ipop_results)

        cmaes_optimizer = CmaEs(POPSIZE, SIGMA0)
        print(cmaes_optimizer.metadata.name)
        cmaes_results = cmaes_optimizer.run_optimization(
            NUM_RUNS, func, BOUNDS, CALL_BUDGET, TOL, num_processes=args.num_processes
        )
        results.append(cmaes_results)

        dump_to_pickle(results, f"006_knn_benchmark_{func.metadata.name}_{DIM}.pkl")
