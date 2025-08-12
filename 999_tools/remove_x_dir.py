"""
Given a list of pkl files, remove x values from all of them to save space.
"""

from optilab.utils.pickle_utils import (
    list_all_pickles,
    load_from_pickle,
    dump_to_pickle,
)
import argparse
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir_path",
        help="Path to directory with pickles.",
        type=Path,
    )
    args = parser.parse_args()

    all_pickles = list_all_pickles(args.dir_path)

    for pickle_path in tqdm(all_pickles, desc='Removing x', unit='file'):
        data = load_from_pickle(pickle_path)
        for run in data:
            run.remove_x()
        dump_to_pickle(data, pickle_path)
