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

                for optimization_run in runs:
                    this_run = optimization_run.stats()

                    print(this_run)
                    
                    this_run = this_run[
                        ["model", "function", "y_median", "y_iqr"]
                    ]

                    print(this_run)

                    this_run = this_run.rename(
                        columns={
                            "y_median": this_run["function"] + "_median",
                            "y_iqr": this_run["function"] + "_iqr",
                        }
                    )

                    print(this_run)
                    
                    this_run.drop(columns=["function"])
                    this_run = this_run.set_index("model").T

                    frames.append(this_run)

    df = pd.concat(frames, ignore_index=True)
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pickle_directory", help="Path to directory containing result pickles."
    )
    args = parser.parse_args()

    result_df = process_pickles(args.pickle_directory)
    result_df.to_csv("aggregated_results.csv")

    print(tabulate(result_df, tablefmt="github"))
