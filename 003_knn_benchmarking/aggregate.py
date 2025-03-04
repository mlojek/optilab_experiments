"""
Aggregate results into a table.
"""

import argparse
import os
import pickle

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm


def process_pickles(directory_path):
    frames = []

    for file_name in tqdm(os.listdir(directory_path)):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(directory_path, file_name)

            with open(file_path, "rb") as f:
                runs = pickle.load(f)
                this_file = []

                for optimization_run in runs:
                    this_run = optimization_run.stats()

                    this_run = this_run[["model", "function", "y_median", "y_iqr"]]

                    this_file.append(this_run)

                this_file = pd.concat(this_file, axis="rows")

                this_file = this_file.rename(
                    columns={
                        "y_median": this_run["function"][0] + "_median",
                        "y_iqr": this_run["function"][0] + "_iqr",
                    }
                )

                this_file = this_file.drop(columns=["function"])
                this_file = this_file.set_index("model").T

                frames.append(this_file)

    df = pd.concat(frames)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pickle_directory", help="Path to directory containing result pickles."
    )
    args = parser.parse_args()

    result_df = process_pickles(args.pickle_directory)
    result_df.to_csv("aggregated_results.csv")

    print(tabulate(result_df, headers=result_df.columns, tablefmt="github"))
