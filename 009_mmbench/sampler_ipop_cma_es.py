"""
CMA-ES variant that samples m, sigma and C from each generation.
"""

from typing import Any, Dict, List

import numpy as np

from optilab.data_classes import Bounds, PointList
from optilab.functions import ObjectiveFunction
from optilab.optimizers.cma_es import CmaEs


class SamplerIpopCmaEs(CmaEs):
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
        current_population_size = self.metadata.population_size
        res_log = PointList(points=[])

        while not self._stop_external(
            res_log,
            current_population_size,
            call_budget,
            target,
            tolerance,
        ):
            es = self._spawn_cmaes(
                bounds,
                function.metadata.dim,
                current_population_size,
                len(bounds) / 2,
            )

            while not self._stop(
                es,
                res_log,
                current_population_size,
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

                for point in solutions:
                    assert point in bounds

                results = PointList(points=[function(x) for x in solutions.points])
                res_log.extend(results)
                x, y = results.pairs()
                es.tell(x, y)

            current_population_size *= 2

        return res_log
