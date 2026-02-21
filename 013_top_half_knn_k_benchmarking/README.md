# Experiment 013: Top-Half KNN K-value Benchmarking

## 1. Introduction

This experiment investigates the effect of the number of nearest neighbors (K) on Top-Half KNN-IPOP-CMA-ES performance. It mirrors experiment 010 but uses the top-half metamodel: each generation, the surrogate estimates all candidates and only the best half is evaluated with the real objective function.

## 2. Setup

- **Benchmark:** CEC 2013 f01–f28
- **Baseline:** IPOP-CMA-ES (no surrogate)
- **Surrogate optimizer:** Top-Half KNN-IPOP-CMA-ES with varying K
- **K values:** dim, 2·dim, 3·dim, 5·dim, 10·dim
- **Buffer size:** 30 · popsize (fixed)
- **Population size:** 4 + floor(3·ln(dim))
- **Budget:** 10⁴ · dim function evaluations
- **Tolerance:** 1e-8
- **Runs per configuration:** 51

## 3. Metrics

- Best objective value found (y)
- Number of function evaluations used

## 4. Usage

```bash
python main.py 2013 10 --num_processes 51
```
