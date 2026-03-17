"""
Benchmarking of IPOP-CMA-ES with Top-Half PolyReg metamodel on CEC2013/CEC2017 benchmark.
"""

import argparse

import numpy as np
from optilab.data_classes import Bounds
from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.optimizers import IpopCmaEs, TopHalfPolyregIpopCmaEs
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
    BUFFER_SIZES = [m * POPSIZE for m in [10, 20, 50, 60, 100]]
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

        for buffer_size in BUFFER_SIZES:
            polyreg_optimizer = TopHalfPolyregIpopCmaEs(POPSIZE, buffer_size)
            print(polyreg_optimizer.metadata.name)
            polyreg_results = polyreg_optimizer.run_optimization(
                NUM_RUNS, func, BOUNDS, CALL_BUDGET, TOL, num_processes=args.num_processes
            )
            polyreg_results.remove_x()
            results.append(polyreg_results)

        dump_to_pickle(results, f"015_th_polyreg_benchmark_{func.metadata.name}_{DIM}.pkl", zstd_compression=None)
