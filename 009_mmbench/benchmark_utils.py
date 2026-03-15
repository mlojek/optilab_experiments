"""
Shared utilities for experiment 009: metamodel benchmarking.
"""

import json
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.stats import spearmanr

from optilab.data_classes import Bounds, Point, PointList
from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.functions.surrogate import (
    KNNSurrogateObjectiveFunction,
    LocallyWeightedPolynomialRegression,
    PolynomialRegression,
    SurrogateObjectiveFunction,
    XGBoostSurrogateObjectiveFunction,
)


BOUNDS = Bounds(-100, 100)


def default_pop_size(dim: int) -> int:
    """
    Default N for train/test splits: 5*dim points each.

    Rationale: reflects typical CMA-ES surrogate training-set sizes
    (~5× the standard population size lambda ≈ dim), and is small
    enough to be computationally tractable even in high dimensions.
    LWR uses all available training points when N < its configured
    neighbour count, which is fine.
    """
    return 5 * dim


def load_dataset(path: str) -> List[Dict]:
    with open(path, "r") as f:
        records = json.load(f)
    for rec in records:
        rec["m"] = np.array(rec["m"], dtype=np.float64)
        rec["C"] = np.array(rec["C"], dtype=np.float64)
    return records


def create_surrogates(
    dim: int, C: np.ndarray
) -> List[Tuple[str, SurrogateObjectiveFunction]]:
    num_neighbors_knn = dim + 2
    num_neighbors_lwr = dim * (dim + 3) // 2 + 2

    lwr = LocallyWeightedPolynomialRegression(
        degree=2,
        num_neighbors=num_neighbors_lwr,
    )
    lwr.set_covariance_matrix(C)

    return [
        ("KNN", KNNSurrogateObjectiveFunction(num_neighbors=num_neighbors_knn)),
        ("LWR", lwr),
        ("PolyReg", PolynomialRegression(degree=2)),
        ("XGBoost", XGBoostSurrogateObjectiveFunction()),
    ]


def _mahalanobis_distances(
    xs: np.ndarray, m: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    """
    Mahalanobis distances of rows of xs from m under cov.

    Uses Cholesky decomposition: d_M(x) = ||L^{-1}(x - m)||
    where cov = L @ L.T.  A small jitter is added for numerical stability.
    Falls back to eigendecomposition for near-singular matrices.
    """
    diff = xs - m  # (n, d)
    jitter = 1e-10 * np.trace(cov) / cov.shape[0]
    try:
        c, low = cho_factor(cov + jitter * np.eye(cov.shape[0]), lower=True)
        # solve L @ z = diff.T  =>  z[:, i] = L^{-1} (x_i - m)
        from scipy.linalg import solve_triangular
        z = solve_triangular(c, diff.T, lower=True)  # (d, n)
        return np.sqrt(np.maximum(0.0, np.sum(z**2, axis=0)))
    except LinAlgError:
        # Fallback via eigendecomposition for (near-)singular cov
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, jitter)
        z = (eigvecs.T @ diff.T) / np.sqrt(eigvals[:, None])  # (d, n)
        return np.sqrt(np.maximum(0.0, np.sum(z**2, axis=0)))


def sample_population(
    m: np.ndarray,
    cov: np.ndarray,
    pop_size: int,
    bounds: Bounds,
    function: CECObjectiveFunction,
) -> PointList:
    """Sample pop_size points from N(m, cov), reflect into bounds, and evaluate."""
    raw_xs = np.random.multivariate_normal(m, cov, size=pop_size)
    points = [function(bounds.reflect(Point(x=x))) for x in raw_xs]
    return PointList(points=points)


def split_population(
    all_points: PointList,
    m: np.ndarray,
    cov: np.ndarray,
    *,
    extrapolation: bool,
) -> Tuple[PointList, PointList]:
    """
    Split a 2N-point population into equal train/test halves.

    Interpolation: random split.
    Extrapolation: sort by Mahalanobis distance; closer half = train, further = test.
    The expected split boundary is sqrt(dim - 2/3) (median of chi(dim)).
    """
    n = len(all_points.points) // 2
    if extrapolation:
        xs = np.array([p.x for p in all_points.points])
        order = np.argsort(_mahalanobis_distances(xs, m, cov))
    else:
        order = np.random.permutation(len(all_points.points))
    train = PointList(points=[all_points.points[i] for i in order[:n]])
    test = PointList(points=[all_points.points[i] for i in order[n:]])
    return train, test


def evaluate_surrogate(
    surrogate: SurrogateObjectiveFunction,
    train_set: PointList,
    test_set: PointList,
) -> Tuple[float, float]:
    """
    Train surrogate on train_set, predict on test_set, return (MAPE, Spearman).
    """
    surrogate.train(train_set)

    y_true = np.array(test_set.y(), dtype=np.float64)
    y_pred = np.array([surrogate(p).y for p in test_set.points], dtype=np.float64)

    eps = 1e-10
    mape = float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))

    if len(y_true) < 3 or np.std(y_true) < eps or np.std(y_pred) < eps:
        spearman = float("nan")
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spearman = float(spearmanr(y_true, y_pred).correlation)

    return mape, spearman
