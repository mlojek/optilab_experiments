"""
Benchmarking of CMA-ES with KNN metamodel on CEC2013 benchmark.
"""

from optilab.data_classes import Bounds
from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.utils import dump_to_pickle
from optilab.optimizers import CmaEs, KnnCmaEs

if __name__ == "__main__":
    # hyperparams:
    DIM = 10
    POPSIZE = DIM * 2
    NUM_NEIGHBORS = [n * POPSIZE for n in [2, 5, 10, 20, 30, 50]]
    NUM_RUNS = 51
    CALL_BUDGET = 1e4 * DIM
    TOL = 1e-8
    SIGMA0 = 1

    # optimized problem
    BOUNDS = Bounds(-100, 100)
    FUNCS = [CECObjectiveFunction(2013, n, DIM) for n in range(1, 29)]
    TARGET = 0.0

    for func in FUNCS:
        results = []

        # optimize using CMA-ES
        cmaes_optimizer = CmaEs(POPSIZE, SIGMA0)
        cmaes_results = cmaes_optimizer.run_optimization(NUM_RUNS, func, BOUNDS, CALL_BUDGET, TOL)

        for num_neigh in NUM_NEIGHBORS:
            print(f'neighbors: {num_neigh}')
            # optimize using KNN-CMA-ES
            knn_optimizer = KnnCmaEs(POPSIZE, SIGMA0, num_neigh)
            knn_results = knn_optimizer.run_optimization(NUM_RUNS, func, BOUNDS, CALL_BUDGET, TOL)
            results.append(knn_results)

        dump_to_pickle([cmaes_results] + results, f'003_knn_benchmark_{func.name}_{DIM}.pkl')
