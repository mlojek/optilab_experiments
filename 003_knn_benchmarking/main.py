"""
Benchmarking of CMA-ES with KNN metamodel on CEC2013/CEC2017 benchmark.
"""

import argparse

from optilab.data_classes import Bounds
from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.utils import dump_to_pickle
from optilab.optimizers import CmaEs, KnnCmaEs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "year",
        type=int,
        choices=[2013, 2017],
        help="Year of CEC benchmark, either 2013 or 2017.",
    )
    parser.add_argument("dim", type=int, help="Dimensionality of benchmark functions.")
    parser.add_argument('--start_from', type=int, default=1, help='Function number to start from.')
    parser.add_argument('--stop_at', type=int, default=100, help='Function number to stop at.')
    args = parser.parse_args()

    # hyperparams:
    DIM = args.dim
    POPSIZE = DIM * 2
    NUM_NEIGHBORS = [m * POPSIZE for m in [2, 5, 10, 20, 30, 50]]
    NUM_RUNS = 51
    CALL_BUDGET = 1e4 * DIM
    TOL = 1e-8
    SIGMA0 = 1

    # optimized problem
    BOUNDS = Bounds(-100, 100)
    FUNCS = {
        2013: [CECObjectiveFunction(2013, n, DIM) for n in range(args.start_from, min(args.stop_at + 1, 29))],
        2017: [CECObjectiveFunction(2017, n, DIM) for n in range(args.start_from, min(args.stop_at + 1, 30))],
    }
    TARGET = 0.0

    for func in FUNCS[args.year]:
        print(func.name)
        results = []

        cmaes_optimizer = CmaEs(POPSIZE, SIGMA0)
        cmaes_results = cmaes_optimizer.run_optimization(
            NUM_RUNS, func, BOUNDS, CALL_BUDGET, TOL
        )
        results.append(cmaes_results)

        for knn_neighbors in NUM_NEIGHBORS:
            knn_optimizer = KnnCmaEs(POPSIZE, SIGMA0, knn_neighbors)
            knn_results = knn_optimizer.run_optimization(
                NUM_RUNS, func, BOUNDS, CALL_BUDGET, TOL
            )
            results.append(knn_results)

        dump_to_pickle(
            results, f"003_knn_benchmark_{func.name}_{DIM}.pkl"
        )
