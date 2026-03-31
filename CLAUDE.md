# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All experiments depend on the `optilab` package, installed in editable mode from `../optilab`. Activate the conda environment before running anything:

```bash
conda activate optilab
```

## Running an Experiment

Each experiment lives in its own numbered directory (`NNN_name/`) and is a standalone Python script:

```bash
cd 017_pdf_interpolation
python main.py --dim 10 --num_runs 51 --num_processes 8
```

Arguments vary per experiment but follow the pattern established in experiment 015:
- `--dim` — dimensionality (10 or 30)
- `--num_runs` — number of independent runs per function (default 51)
- `--num_processes` — parallel workers via `multiprocessing.Pool`
- `--start_from` / `--stop_at` — subset of CEC functions to run

## Repository Structure

- `000_template/README.md` — template for experiment writeups (Introduction, Algorithm, Evaluation, Results, Discussion, Related Work, Conclusion)
- `NNN_name/` — each experiment directory; most contain only `main.py` plus output files (`.pkl`, `.png`, `.csv`)
- `999_tools/` — shared utilities: `compress_pickle.py`, `remove_x_dir.py`, `recreate_summary.py`

## optilab Package Architecture

The `optilab` package (at `../optilab/src/optilab/`) provides all core abstractions:

**Optimizers** (`optilab.optimizers`): All inherit from `Optimizer`. The base `optimize()` method runs one optimization and returns a `PointList`. `run_optimization(num_runs, ...)` parallelises over runs via `multiprocessing.Pool` and returns an `OptimizationRun`. Key implementations:
- `CmaEs` — wraps PyCMA 4.0.0; exposes `_spawn_cmaes()` and `_stop_internal()` as static methods for subclassing
- `IpopCmaEs` — IPOP restarts with doubling population; sigma0 = `len(bounds) / 2`

**Data classes** (`optilab.data_classes`):
- `Bounds(lower, upper)` — `len(bounds)` returns width; `bounds.to_list()` for PyCMA; `bounds.random_point(dim)` for init
- `PointList` — list of `Point(x, y)`; `from_list(xs)` from raw numpy arrays; `.pairs()` returns `(x_list, y_list)` for `es.tell()`; `.best_y()` raises on empty list (guard with `len(log) > 0`)
- `OptimizationRun` — wraps metadata + `List[PointList]` logs; `.remove_x()` strips x values to save memory before pickling

**Benchmarks** (`optilab.functions.benchmarks`):
- `CECObjectiveFunction(year, function_num, dim)` — CEC2013 (1–28) or CEC2017 (1–29) via `opfunu`; returns `f(x) - f_global`

**Utilities** (`optilab.utils`):
- `dump_to_pickle(data, path, zstd_compression=1)` — pass `zstd_compression=None` to skip compression
- `load_from_pickle(path)` — auto-detects `.zstd.pkl` extension

**Plotting** (`optilab.plotting`): `plot_box_plot`, `plot_convergence_curve`, `plot_ecdf_curves` — used in post-processing scripts, not usually in `main.py`.

## Experiment Conventions

- Standard hyperparams: `POPSIZE = int(4 + np.floor(3 * np.log(DIM)))`, `CALL_BUDGET = int(1e4 * DIM)`, `TOL = 1e-8`
- CEC2017 bounds: `Bounds(-100, 100)`
- Output files named `NNN_description_{func_name}_{DIM}D.pkl` / `.png`
- When subclassing or extending optimizers, prefer creating a standalone function in the experiment directory rather than modifying optilab
- When custom IPOP loops are needed, use `CmaEs._spawn_cmaes()`, `CmaEs._stop_internal()`, and `Optimizer._stop_budget()` / `Optimizer._stop_target_found()` as static helpers instead of reimplementing them
