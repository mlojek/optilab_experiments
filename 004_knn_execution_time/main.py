"""
Measuring execution time of CMA-ES and KNN-CMA-ES on sphere funciton in increasing dimensionality.
"""

import time
import json
from optilab.data_classes import Bounds
from optilab.functions.unimodal import SphereFunction
from optilab.utils import dump_to_pickle
from optilab.optimizers import CmaEs, KnnCmaEs

if __name__ == "__main__":
    # hyperparams:
    DIMS = [2**i for i in range(12)]
    NUM_RUNS = 11
    TOL = 1e-8
    SIGMA0 = 1

    results = []
    times = {'dim': [], 'cma-es': [], 'knn-cma-es': []}

    # optimized problem
    BOUNDS = Bounds(-100, 100)

    for dim in DIMS:
        print(f'Dimensionality: {dim}')

        cmaes_start = time.time()
        cmaes_optimizer = CmaEs(dim*2, SIGMA0)
        cmaes_results = cmaes_optimizer.run_optimization(
            NUM_RUNS, SphereFunction(dim), BOUNDS, dim*1e4, TOL
        )
        cmaes_stop = time.time()

        results.append(cmaes_results)

        knn_start = time.time()
        knn_optimizer = KnnCmaEs(dim*2, SIGMA0, dim+2, dim*10)
        knn_results = knn_optimizer.run_optimization(
            NUM_RUNS, SphereFunction(dim), BOUNDS, dim*1e4, TOL
        )
        knn_stop = time.time()

        results.append(knn_results)

        times['dim'].append(dim)
        times['cma-es'].append((cmaes_stop - cmaes_start)/NUM_RUNS)
        times['knn-cma-es'].append((knn_stop - knn_start)/NUM_RUNS)

    dump_to_pickle(
        results, f"004_knn_execution_time.pkl"
    )

    with open('time.json', 'w') as file_handle:
        json.dump(times, file_handle, indent=4)
