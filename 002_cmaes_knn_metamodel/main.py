"""
Benchmarking of CMA-ES with KNN metamodel
"""

from optilab.data_classes import Bounds
from optilab.functions.unimodal import SphereFunction
from optilab.utils import dump_to_pickle
from optilab.optimizers import CmaEs, KnnCmaEs

if __name__ == "__main__":
    # hyperparams:
    DIM = 2
    POPSIZE = DIM * 4
    NUM_NEIGHBORS = POPSIZE * 5
    NUM_RUNS = 51
    CALL_BUDGET = 1e4 * DIM
    TOL = 1e-8
    SIGMA0 = 1

    # optimized problem
    BOUNDS = Bounds(-100, 100)
    FUNC = SphereFunction(DIM)
    TARGET = 0.0

    # optimize using CMA-ES
    cmaes_optimizer = CmaEs(POPSIZE, SIGMA0)
    cmaes_results = cmaes_optimizer.run_optimization(NUM_RUNS, FUNC, BOUNDS, CALL_BUDGET, TOL)

    # optimize using KNN-CMA-ES
    knn_optimizer = KnnCmaEs(POPSIZE, SIGMA0, NUM_NEIGHBORS)
    knn_results = knn_optimizer.run_optimization(NUM_RUNS, FUNC, BOUNDS, CALL_BUDGET, TOL)

    dump_to_pickle([cmaes_results, knn_results], f'002_knn_cma_es_{FUNC.name}_{DIM}.pkl')
