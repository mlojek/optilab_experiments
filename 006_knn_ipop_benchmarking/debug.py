"""
Debug version of main script for profiling the optimizers.
"""

import argparse

import numpy as np
from optilab.data_classes import Bounds
from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.optimizers import KnnIpopCmaEs


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
        "function",
        type=int,
        help="Number of the function to optimize.",
    )
    args = parser.parse_args()

    # optimized problem
    DIM = args.dim
    BOUNDS = Bounds(-100, 100)
    TARGET = 0.0
    FUNC = CECObjectiveFunction(args.year, args.function, DIM)

    # hyperparams:
    POPSIZE = int(4 + np.floor(3 * np.log(DIM)))
    NUM_NEIGHBORS = DIM + 2
    BUFFER_SIZE = 5 * POPSIZE
    CALL_BUDGET = 1e4 * DIM
    TOL = 1e-8

    # optimization execution
    knn_optimizer = KnnIpopCmaEs(POPSIZE, NUM_NEIGHBORS, BUFFER_SIZE)
    print(knn_optimizer.metadata.name)
    knn_results = knn_optimizer.optimize(FUNC, BOUNDS, CALL_BUDGET, TOL, TARGET)