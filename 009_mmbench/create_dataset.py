import argparse
import json
from functools import partial
from multiprocessing.pool import Pool
from typing import Dict, List

import numpy as np

from optilab.functions.benchmarks import CECObjectiveFunction
from optilab.data_classes import Bounds
from sampler_ipop_cma_es import SamplerIpopCmaEs


def collect_for_function(
    function_num: int,
    dim: int,
    popsize: int,
    sigma0: float,
    call_budget: float,
    tol: float,
    num_samples: int,
) -> List[Dict]:
    """
    Run CMA-ES on one CEC 2013 function and sample representative states.

    Args:
        function_num: CEC 2013 function number (1-28).
        dim: Dimensionality.
        popsize: Population size.
        sigma0: Initial sigma.
        call_budget: Maximum function evaluations.
        tol: Tolerance.
        num_samples: Number of (m, sigma, C) tuples to sample.

    Returns:
        List of dicts with function_num, dim, m, sigma, C.
    """
    bounds = Bounds(-100, 100)
    func = CECObjectiveFunction(2013, function_num, dim)

    optimizer = SamplerIpopCmaEs(popsize, sigma0)
    optimizer.optimize(func, bounds, int(call_budget), tol)

    states = optimizer.collected_states

    assert len(states) > num_samples

    if len(states) >= num_samples:
        indices = np.linspace(0, len(states) - 1, num_samples, dtype=int)
        selected = [states[i] for i in indices]
    else:
        # Sample with replacement to always reach num_samples
        indices = np.random.choice(len(states), size=num_samples, replace=True)
        selected = [states[i] for i in indices]

    records = []
    for state in selected:
        records.append(
            {
                "function_num": function_num,
                "dim": dim,
                "m": state["m"].tolist(),
                "sigma": state["sigma"],
                "C": state["C"].tolist(),
            }
        )

    print(
        f"  f{function_num:02d} dim={dim}: {len(states)} generations, "
        f"sampled {len(selected)} states"
    )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create metamodel benchmark dataset from CMA-ES runs on CEC 2013."
    )
    parser.add_argument(
        "dim",
        type=int,
        help="Dimensionality (e.g. 10 or 30).",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=1,
        help="First function number (default: 1).",
    )
    parser.add_argument(
        "--stop_at",
        type=int,
        default=28,
        help="Last function number (default: 28).",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of (m, sigma, C) samples per function (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: dataset_{dim}d.json).",
    )
    args = parser.parse_args()

    DIM = args.dim
    POPSIZE = 50
    SIGMA0 = 1.0
    CALL_BUDGET = 1e4 * DIM
    TOL = 1e-8

    output_path = args.output or f"dataset_{DIM}d.json"
    func_nums = list(range(args.start_from, min(args.stop_at + 1, 29)))

    print(
        f"Creating dataset: dim={DIM}, functions={func_nums}, "
        f"samples_per_func={args.num_samples}"
    )

    worker = partial(
        collect_for_function,
        dim=DIM,
        popsize=POPSIZE,
        sigma0=SIGMA0,
        call_budget=CALL_BUDGET,
        tol=TOL,
        num_samples=args.num_samples,
    )

    all_records: List[Dict] = []

    if args.num_processes > 1:
        with Pool(processes=args.num_processes) as pool:
            results = pool.map(worker, func_nums)
        for batch in results:
            all_records.extend(batch)
    else:
        for fn in func_nums:
            batch = worker(fn)
            all_records.extend(batch)

    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=2)

    print(f"\nSaved {len(all_records)} records to {output_path}")
