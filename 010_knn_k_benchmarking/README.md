# Experiment 010: KNN K-value Benchmarking

## 1. Introduction

This experiment investigates the effect of the number of nearest neighbors (K) on KNN-IPOP-CMA-ES performance. In experiment 006 we varied the buffer size with a fixed K=dim+2. Here we fix the buffer size at 30·popsize and sweep over different K values to find the optimal neighborhood size.

## 2. Setup

- **Benchmark:** CEC 2013 f01–f28
- **Baseline:** IPOP-CMA-ES (no surrogate)
- **Surrogate optimizer:** KNN-IPOP-CMA-ES with varying K
- **K values:** 3, 5, dim+2, 2·dim, 3·dim, 5·dim
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
