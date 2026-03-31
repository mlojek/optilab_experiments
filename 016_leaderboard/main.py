import argparse
import json

from dataclasses import asdict, dataclass
from pathlib import Path
from optilab.utils.pickle_utils import load_from_pickle


@dataclass
class OptimizationRunBrief:
    function_name: str
    dim: int
    optimizer_name: str
    ys: list[float]
    evals: list[float]


def convert(pkl_path: Path) -> None:
    runs = load_from_pickle(pkl_path)

    briefs = [
        OptimizationRunBrief(
            function_name=run.function_metadata.name,
            dim=run.function_metadata.dim,
            optimizer_name=run.model_metadata.name,
            ys=run.bests_y(),
            evals=run.log_lengths(),
        )
        for run in runs
    ]

    out_path = pkl_path.with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump([asdict(b) for b in briefs], f, indent=4)

    print(f"{pkl_path.name} -> {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    for pkl_path in sorted(args.directory.glob("*.pkl")):
        convert(pkl_path)
