"""
CMA-ES variant that samples m, sigma and C from each generation.
"""

from typing import Any, Dict, List

import numpy as np

from optilab.data_classes import Bounds, PointList
from optilab.functions import ObjectiveFunction
from optilab.optimizers.cma_es import CmaEs


class SamplerCmaEs(CmaEs):
    """
    CMA-ES variant that records (m, sigma, C) at each generation.
    """

    def __init__(self, population_size: int, sigma0: float):
        super().__init__(population_size, sigma0)
        self.collected_states: List[Dict[str, Any]] = []

    def optimize(
        self,
        function: ObjectiveFunction,
        bounds: Bounds,
        call_budget: int,
        tolerance: float,
        target: float = 0.0,
    ) -> PointList:
        """
        Run optimization while collecting CMA-ES internal state each generation.
        """
        es = self._spawn_cmaes(
            bounds,
            function.metadata.dim,
            self.metadata.population_size,
            self.metadata.hyperparameters["sigma0"],
        )

        res_log = PointList(points=[])
        self.collected_states = []

        while not self._stop(
            es,
            res_log,
            self.metadata.population_size,
            call_budget,
            target,
            tolerance,
        ):
            self.collected_states.append(
                {
                    "m": np.array(es.mean).copy(),
                    "sigma": float(es.sigma),
                    "C": np.array(es.C).copy(),
                }
            )

            solutions = PointList.from_list(es.ask())
            results = PointList(points=[function(x) for x in solutions.points])
            res_log.extend(results)
            x, y = results.pairs()
            es.tell(x, y)

        return res_log
