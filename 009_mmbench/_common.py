"""
Shared utilities for experiment 009: metamodel benchmarking.
"""

import json
from typing import Dict, List, Tuple

import numpy as np
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


def load_dataset(path: str) -> List[Dict]:
    """
    Load dataset JSON and convert m/C back to numpy arrays.

    Args:
        path: Path to the JSON dataset file.

    Returns:
        List of dicts with keys: function_num, dim, m, sigma, C.
    """
    with open(path, "r") as f:
        records = json.load(f)

    for rec in records:
        rec["m"] = np.array(rec["m"], dtype=np.float64)
        rec["C"] = np.array(rec["C"], dtype=np.float64)

    return records


def create_surrogates(
    dim: int, C: np.ndarray
) -> List[Tuple[str, SurrogateObjectiveFunction]]:
    """
    Create the three surrogate functions to benchmark.

    Args:
        dim: Dimensionality of the problem.
        C: Covariance matrix (passed to LWR for realistic usage).

    Returns:
        List of (name, surrogate) tuples.
    """
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


def sample_population(
    m: np.ndarray,
    cov: np.ndarray,
    pop_size: int,
    bounds: Bounds,
    function: CECObjectiveFunction,
) -> PointList:
    """
    Sample a population from N(m, cov), reflect into bounds, and evaluate
    with the real objective function.

    Args:
        m: Mean vector.
        cov: Full covariance matrix (sigma^2 * C).
        pop_size: Number of points to sample.
        bounds: Search space bounds.
        function: The real objective function to evaluate points.

    Returns:
        PointList of evaluated points.
    """
    raw_xs = np.random.multivariate_normal(m, cov, size=pop_size)
    points = []
    for x in raw_xs:
        p = Point(x=x)
        p = bounds.reflect(p)
        p = function(p)
        points.append(p)
    return PointList(points=points)


def evaluate_surrogate(
    surrogate: SurrogateObjectiveFunction,
    train_set: PointList,
    test_set: PointList,
) -> Tuple[float, float]:
    """
    Train a surrogate on train_set, predict on test_set, and compute metrics.

    Args:
        surrogate: The surrogate function to evaluate.
        train_set: Training data (evaluated PointList).
        test_set: Test data (evaluated PointList).

    Returns:
        Tuple of (MAPE, Spearman rank correlation).
    """
    surrogate.train(train_set)

    y_true = np.array(test_set.y(), dtype=np.float64)
    y_pred = np.array(
        [surrogate(p).y for p in test_set.points], dtype=np.float64
    )

    # MAPE: Mean Absolute Percentage Error
    eps = 1e-10
    mape = float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))

    # Spearman rank correlation
    if len(y_true) < 3 or np.std(y_true) < eps or np.std(y_pred) < eps:
        spearman = float("nan")
    else:
        spearman = float(spearmanr(y_true, y_pred).correlation)

    return mape, spearman
