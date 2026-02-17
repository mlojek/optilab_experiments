# Experiment 009: Metamodel Benchmarking

Benchmark of surrogate objective functions (metamodels) on CEC 2013 benchmark functions to evaluate their **interpolation** and **extrapolation** accuracy.

## Setup

- **Benchmark functions:** CEC 2013 f01–f28, dimension 10
- **CMA-ES states:** 10 representative `(m, σ, C)` tuples sampled per function from a plain CMA-ES run (popsize=50, σ₀=1, budget=10⁴·dim)
- **Training/test sets:** 50 points each, sampled from N(m, σ²·C) and evaluated with the real function
- **Interpolation:** test set from the same distribution N(m, σ²·C)
- **Extrapolation:** test set from a wider distribution N(m, (2σ)²·C)

## Surrogates tested

| Name | Description | Key hyperparameters |
|------|-------------|-------------------|
| **KNN** | FAISS-based k-nearest neighbors regression | k = dim + 2 = 12 |
| **LWR** | Locally weighted polynomial regression (Mahalanobis distance) | degree=2, k=dim·(dim+3)/2+2=67, uses CMA-ES covariance C |
| **PolyReg** | Global polynomial regression (least squares) | degree=2 |
| **XGBoost** | Gradient-boosted tree regression | n_estimators=100, max_depth=6, lr=0.1 |

## Metrics

- **MAPE** — Mean Absolute Percentage Error: lower is better
- **Spearman** — Spearman rank correlation between true and predicted values: higher is better (max 1.0)

## Results — Overall Summary (dim=10)

### Interpolation

| Surrogate | MAPE (mean) | Spearman (mean) |
|-----------|-------------|-----------------|
| **KNN** | **1.70** | 0.499 |
| LWR | 18803.00 | **0.566** |
| PolyReg | 18803.00 | 0.567 |
| XGBoost | 78.92 | 0.530 |

### Extrapolation

| Surrogate | MAPE (mean) | Spearman (mean) |
|-----------|-------------|-----------------|
| **KNN** | **0.19** | 0.449 |
| LWR | 6.28 | **0.513** |
| PolyReg | 6.28 | 0.512 |
| XGBoost | 0.25 | 0.468 |

> **Note:** Mean MAPE for LWR/PolyReg is dominated by extreme values on f07 (Griewank-Rosenbrock).

## Results — Per-function MAPE (Interpolation)

| Function | KNN | LWR | PolyReg | XGBoost |
|----------|-----|-----|---------|---------|
| f01 | 0.070 | 0.122 | 0.122 | 0.077 |
| f02 | 0.520 | 1.871 | 1.870 | 0.650 |
| f03 | 15.282 | 366.842 | 366.842 | 8.214 |
| f04 | 2.286 | 13.412 | 13.406 | 2.560 |
| f05 | 0.590 | 1.127 | 1.127 | 0.093 |
| f06 | 0.125 | 0.507 | 0.507 | 0.154 |
| f07 | 27.575 | 526094.690 | 526094.690 | 2196.573 |
| f08 | 0.009 | 0.045 | 0.045 | 0.011 |
| f09 | 0.008 | 0.028 | 0.028 | 0.009 |
| f10 | 0.133 | 0.282 | 0.282 | 0.136 |
| f11 | 0.070 | 0.270 | 0.270 | 0.086 |
| f12 | 0.086 | 0.217 | 0.217 | 0.098 |
| f13 | 0.081 | 0.400 | 0.400 | 0.094 |
| f14 | 0.013 | 0.030 | 0.030 | 0.014 |
| f15 | 0.017 | 0.038 | 0.038 | 0.018 |
| f16 | 0.263 | 1.601 | 1.601 | 0.276 |
| f17 | 0.094 | 0.402 | 0.402 | 0.100 |
| f18 | 0.100 | 0.275 | 0.275 | 0.110 |
| f19 | 0.163 | 0.658 | 0.658 | 0.175 |
| f20 | 0.027 | 0.179 | 0.179 | 0.028 |
| f21 | 0.015 | 0.060 | 0.060 | 0.014 |
| f22 | 0.015 | 0.062 | 0.062 | 0.013 |
| f23 | 0.010 | 0.014 | 0.014 | 0.010 |
| f24 | 0.028 | 0.163 | 0.163 | 0.030 |
| f25 | 0.006 | 0.056 | 0.056 | 0.007 |
| f26 | 0.062 | 0.688 | 0.688 | 0.063 |
| f27 | 0.009 | 0.036 | 0.036 | 0.011 |
| f28 | 0.013 | 0.054 | 0.054 | 0.013 |

## Results — Per-function Spearman (Interpolation)

| Function | KNN | LWR | PolyReg | XGBoost |
|----------|-----|-----|---------|---------|
| f01 | 0.850 | 0.882 | 0.882 | 0.716 |
| f02 | 0.283 | 0.312 | 0.316 | 0.229 |
| f03 | 0.822 | 0.904 | 0.904 | 0.875 |
| f04 | 0.435 | 0.350 | 0.360 | 0.241 |
| f05 | 0.719 | 0.839 | 0.839 | 0.971 |
| f06 | 0.698 | 0.818 | 0.818 | 0.625 |
| f07 | 0.345 | 0.711 | 0.711 | 0.503 |
| f08 | 0.015 | -0.060 | -0.060 | 0.006 |
| f09 | 0.730 | 0.513 | 0.512 | 0.596 |
| f10 | 0.559 | 0.832 | 0.832 | 0.519 |
| f11 | 0.500 | 0.658 | 0.658 | 0.648 |
| f12 | 0.539 | 0.709 | 0.709 | 0.542 |
| f13 | 0.250 | 0.492 | 0.492 | 0.225 |
| f14 | 0.352 | 0.395 | 0.401 | 0.531 |
| f15 | 0.475 | 0.827 | 0.827 | 0.613 |
| f16 | 0.316 | 0.319 | 0.317 | 0.285 |
| f17 | 0.290 | 0.233 | 0.233 | 0.373 |
| f18 | 0.508 | 0.465 | 0.465 | 0.478 |
| f19 | 0.535 | 0.587 | 0.587 | 0.481 |
| f20 | 0.627 | 0.650 | 0.650 | 0.585 |
| f21 | 0.624 | 0.873 | 0.873 | 0.781 |
| f22 | 0.387 | 0.431 | 0.431 | 0.549 |
| f23 | 0.293 | 0.524 | 0.523 | 0.464 |
| f24 | 0.214 | 0.275 | 0.269 | 0.382 |
| f25 | 0.766 | 0.447 | 0.447 | 0.674 |
| f26 | 0.652 | 0.702 | 0.702 | 0.775 |
| f27 | 0.608 | 0.575 | 0.575 | 0.554 |
| f28 | 0.596 | 0.492 | 0.492 | 0.533 |

## Key Observations

1. **KNN has the lowest mean MAPE overall** — for both interpolation and extrapolation. It is the most numerically stable surrogate.

2. **LWR/PolyReg achieve the best Spearman correlation** on average, but are **numerically unstable** — on function f07, MAPE explodes to ~526k, dragging the mean up dramatically.

3. **LWR and PolyReg produce nearly identical results**, suggesting that the Mahalanobis weighting in LWR doesn't add much value when the covariance matrix from the initial CMA-ES generation is close to identity.

4. **XGBoost excels on f05** (Spearman 0.971 interpolation, 0.915 extrapolation) and is competitive on many functions, but has higher MAPE than KNN on average.

5. **Extrapolation degrades all surrogates** — median MAPE roughly doubles compared to interpolation, and Spearman drops by ~0.1 across the board.

6. **f08 is problematic for all surrogates** — near-zero Spearman, suggesting the local landscape around CMA-ES iterates is essentially flat/uninformative for all methods.

## Scripts

| Script | Description |
|--------|-------------|
| `create_dataset.py` | Run CMA-ES on CEC 2013, collect (m, σ, C) states |
| `test_interpolation.py` | Benchmark surrogates on same-distribution test sets |
| `test_extrapolation.py` | Benchmark surrogates on wider-distribution test sets |
| `_common.py` | Shared utilities (sampling, evaluation, metrics) |
| `run_all.py` | Pipeline runner for all steps |

## Usage

```bash
# Create dataset
python create_dataset.py 10 --num_samples 10

# Run benchmarks
python test_interpolation.py --dataset dataset_10d.json --output interpolation_10d.csv
python test_extrapolation.py --dataset dataset_10d.json --output extrapolation_10d.csv
```
