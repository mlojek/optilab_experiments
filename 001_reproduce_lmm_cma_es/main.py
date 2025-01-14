"""
Reproducing LMM-CMA-ES optimizer results.
"""

from optilab.data_classes import Bounds
from optilab.functions.unimodal import CumulativeSquaredSums
from optilab.utils import dump_to_pickle
from optilab.optimizers import CmaEs, LmmCmaEs

if __name__ == "__main__":
    # hyperparams:
    DIM = 2
    POPSIZE = 6
    NUM_RUNS = 51
    CALL_BUDGET = 1e4 * DIM
    TOL = 1e-10
    SIGMA0 = 1

    # optimized problem
    BOUNDS = Bounds(-10, 10)
    FUNC = CumulativeSquaredSums(DIM)
    TARGET = 0.0

    # optimize using CMA-ES
    cmaes_optimizer = CmaEs(POPSIZE, SIGMA0)
    cmaes_runs = cmaes_optimizer.run_optimization(NUM_RUNS, FUNC, BOUNDS, CALL_BUDGET, TOL, TARGET)

    # optimize using LMM-CMA-ES
    lmm_optimizer = LmmCmaEs(POPSIZE, SIGMA0, 2)
    lmm_runs = lmm_optimizer.run_optimization(NUM_RUNS, FUNC, BOUNDS, CALL_BUDGET, TOL, TARGET)

    # save results to pickle
    dump_to_pickle([cmaes_runs, lmm_runs], f'001_reproduce_lmm_cma_es_{FUNC.name}_{DIM}.pkl')
