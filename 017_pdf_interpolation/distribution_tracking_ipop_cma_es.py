"""
Instrumented IPOP-CMA-ES that tracks interpolation/extrapolation per generation.

For each generation, every newly sampled point is classified as:
- Interpolatable: normalised PDF >= threshold under at least one previous generation's
  distribution, where normalised PDF = exp(-0.5 * Mahalanobis²) ∈ (0, 1].
- Extrapolation: normalised PDF < threshold under all previous distributions.

Using the normalised PDF (ratio to the peak density) removes the dimension- and
sigma-dependent scaling of the raw PDF, making the threshold meaningful regardless
of search space size or convergence stage.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from optilab.data_classes import Bounds, PointList
from optilab.functions import ObjectiveFunction
from optilab.optimizers.cma_es import CmaEs
from optilab.optimizers.optimizer import Optimizer


@dataclass
class DistributionSnapshot:
    """Parameters of the multivariate Gaussian used to generate one generation.

    Stores the mean and pre-computed precision matrix (inverse covariance) so that
    normalised-PDF checks are a single matrix-vector multiply rather than a full
    Cholesky decomposition on every call.
    """
    mean: np.ndarray
    precision: np.ndarray  # (sigma^2 * C)^{-1}

    @classmethod
    def from_cmaes(cls, es) -> "DistributionSnapshot":
        cov = (es.sigma ** 2) * es.C
        precision = np.linalg.inv(cov)
        return cls(mean=es.mean.copy(), precision=precision)


@dataclass
class GenerationRecord:
    """Interpolation/extrapolation counts for a single generation."""
    n_interpolated: int
    n_extrapolated: int

    @property
    def total(self) -> int:
        return self.n_interpolated + self.n_extrapolated

    @property
    def pct_interpolated(self) -> float:
        return self.n_interpolated / self.total * 100 if self.total > 0 else 0.0


def _is_interpolatable(
    x: np.ndarray,
    history: List[DistributionSnapshot],
    log_threshold: float,
) -> bool:
    """Return True if the normalised PDF of x is >= threshold under any snapshot.

    normalised_pdf(x) = exp(-0.5 * (x-m)^T Σ^{-1} (x-m))

    Equivalently, checks whether -0.5 * Mahalanobis² >= log(threshold).
    """
    for snap in history:
        d = x - snap.mean
        maha2 = d @ snap.precision @ d
        if -0.5 * maha2 >= log_threshold:
            return True
    return False


def _stop(
    log: PointList,
    pop_size: int,
    call_budget: int,
    target: float,
    tolerance: float,
) -> bool:
    return Optimizer._stop_budget(log, pop_size, call_budget) or (
        len(log) > 0 and Optimizer._stop_target_found(log, target, tolerance)
    )


def run_distribution_tracking_ipop(
    function: ObjectiveFunction,
    bounds: Bounds,
    call_budget: int,
    tolerance: float,
    population_size: Optional[int] = None,
    pdf_threshold: float = 0.005,
    target: float = 0.0,
) -> List[GenerationRecord]:
    """
    Run IPOP-CMA-ES while recording interpolation/extrapolation per generation.

    Args:
        function: Objective function to optimise.
        bounds: Search space bounds.
        call_budget: Maximum number of function evaluations.
        tolerance: Convergence tolerance.
        population_size: Initial population size (default: 4 + floor(3*ln(dim))).
        pdf_threshold: Normalised-PDF threshold in (0, 1] above which a point is
            considered interpolatable. Equivalent to a Mahalanobis distance cutoff
            of sqrt(-2 * ln(threshold)) — e.g. 0.005 ≈ 3.26 std deviations.
        target: Global optimum value (default 0.0).

    Returns:
        List of GenerationRecord, one entry per generation across all IPOP restarts.
    """
    dim = function.metadata.dim
    if population_size is None:
        population_size = int(4 + np.floor(3 * np.log(dim)))

    log_threshold = np.log(pdf_threshold)  # compute once; used as -0.5*maha² cutoff
    res_log = PointList(points=[])
    generation_records: List[GenerationRecord] = []
    distribution_history: List[DistributionSnapshot] = []
    current_pop_size = population_size

    while not _stop(res_log, current_pop_size, call_budget, target, tolerance):
        es = CmaEs._spawn_cmaes(bounds, dim, current_pop_size, len(bounds) / 2)

        while not (
            CmaEs._stop_internal(es)
            or _stop(res_log, current_pop_size, call_budget, target, tolerance)
        ):
            # 1. Snapshot the distribution that will generate this generation's points.
            #    Pre-compute precision matrix so classification is O(d²) per point.
            snapshot = DistributionSnapshot.from_cmaes(es)

            # 2. Sample the generation.
            solutions_x = es.ask()

            # 3. Classify each point against all PREVIOUS distributions.
            n_interpolated = sum(
                _is_interpolatable(x, distribution_history, log_threshold)
                for x in solutions_x
            )
            generation_records.append(
                GenerationRecord(n_interpolated, len(solutions_x) - n_interpolated)
            )

            # 4. Evaluate and update CMA-ES.
            solutions = PointList.from_list(solutions_x)
            results = PointList(points=[function(pt) for pt in solutions.points])
            res_log.extend(results)
            x_vals, y_vals = results.pairs()
            es.tell(x_vals, y_vals)

            # 5. Add the snapshot to history so future generations can check against it.
            distribution_history.append(snapshot)

        current_pop_size *= 2

    return generation_records
