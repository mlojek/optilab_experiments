'''
Recreate the summary.md file from all the csv files the main optilab script produced.
'''

import argparse
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from optilab.utils.stat_test import display_test_grid
from optilab.utils.aggregate_pvalues import aggregate_pvalues
from optilab.utils.aggregate_stats import aggregate_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stats_dir', type=Path, help='Path to the directory with stats CSVs.',)
    parser.add_argument('y_pvalues_dir', type=Path, help='Path to the directory with y pvalues CSVs.',)
    parser.add_argument('evals_pvalues_dir', type=Path, help='Path to the directory with evals pvalues CSVs.',)
    parser.add_argument('--significance', type=float, default=0.05, help='Significance level for p-value aggregation.',)
    args = parser.parse_args()

    # from stats dir read all csvs and read them into a dictionary where key is the filename without extensions and value is the dataframe
    stats_dfs = {}
    for csv_file in args.stats_dir.glob('*.csv'):
        df_name = csv_file.stem.split('.')[0]  # filename without extension
        stats_dfs[df_name] = pd.read_csv(csv_file)
        # drop first unnamed column if exists
        if 'Unnamed: 0' in stats_dfs[df_name].columns:
            stats_dfs[df_name] = stats_dfs[df_name].drop(columns=['Unnamed: 0'])

    # read y pvalues
    y_pvalues_dfs = {}
    for csv_file in args.y_pvalues_dir.glob('*.csv'):
        df_name = csv_file.stem.split('.')[0]  # filename without extension
        y_pvalues_dfs[df_name] = pd.read_csv(csv_file)
        # drop first unnamed column if exists
        if 'Unnamed: 0' in y_pvalues_dfs[df_name].columns:
            y_pvalues_dfs[df_name] = y_pvalues_dfs[df_name].drop(columns=['Unnamed: 0'])

    # read evals pvalues
    evals_pvalues_dfs = {}
    for csv_file in args.evals_pvalues_dir.glob('*.csv'):
        df_name = csv_file.stem.split('.')[0]  # filename without extension
        evals_pvalues_dfs[df_name] = pd.read_csv(csv_file)
        # drop first unnamed column if exists
        if 'Unnamed: 0' in evals_pvalues_dfs[df_name].columns:
            evals_pvalues_dfs[df_name] = evals_pvalues_dfs[df_name].drop(columns=['Unnamed: 0'])

    # get the intersection of all keys
    common_keys = set(stats_dfs.keys()) & set(y_pvalues_dfs.keys()) & set(evals_pvalues_dfs.keys())
    
    # join all the dataframes like this: {key: (stats_df, y_pvalues_df, evals_pvalues_df)
    combined_dfs = {k: (stats_dfs[k], y_pvalues_dfs[k], evals_pvalues_dfs[k]) for k in sorted(common_keys)}

    # aggregation dataframes
    stats_to_aggregate_df = pd.DataFrame(
        columns=["model", "function", "y_median", "y_iqr"]
    )
    y_pvalues_to_aggregate_df = pd.DataFrame(
        columns=["model", "function", "alternative", "pvalue"]
    )
    evals_pvalues_to_aggregate_df = pd.DataFrame(
        columns=["model", "function", "alternative", "pvalue"]
    )

    for key, (stats_df, y_pvalues_df, evals_pvalues_df) in combined_dfs.items():
        print(f"# File {key}")
        
        # add stats to aggregation
        stats_to_concat = pd.DataFrame(stats_df, columns=stats_to_aggregate_df.columns)
        stats_to_aggregate_df = pd.concat(
            [stats_to_aggregate_df, stats_to_concat], axis=0
        )

        # print stats
        stats_evals = stats_df.filter(like="evals_", axis=1)
        stats_y = stats_df.filter(like="y_", axis=1)
        stats_df = stats_df.drop(columns=stats_evals.columns.union(stats_y.columns))

        print(tabulate(stats_df, headers="keys", tablefmt="github"), "\n")
        print(tabulate(stats_y, headers="keys", tablefmt="github"), "\n")
        print(tabulate(stats_evals, headers="keys", tablefmt="github"), "\n")


        # convert y_pvalues_df to a list of list of float
        pvalues_y = y_pvalues_df.values.tolist()

        # print y pvalues
        print("## Mann Whitney U test on optimization results (y).")
        print("p-values for alternative hypothesis row < column")
        print(display_test_grid(pvalues_y), "\n")

        # append y pvalues to aggregation df
        better_df = pd.DataFrame(
            [
                {
                    "model": stats.model,
                    "function": stats.function,
                    "alternative": "better",
                    "pvalue": row[0],
                }
                for row, (_, stats) in zip(
                    pvalues_y[1:], list(stats_df.iterrows())[1:]
                )
            ]
        )
        worse_df = pd.DataFrame(
            [
                {
                    "model": stats.model,
                    "function": stats.function,
                    "alternative": "worse",
                    "pvalue": pvalue,
                }
                for pvalue, (_, stats) in zip(
                    pvalues_y[0][1:], list(stats_df.iterrows())[1:]
                )
            ]
        )
        y_pvalues_to_aggregate_df = pd.concat(
            [y_pvalues_to_aggregate_df, better_df, worse_df], axis=0
        )

        # convert evals_pvalues_df to a list of list of float
        pvalues_evals = evals_pvalues_df.values.tolist()

        # print evals pvalues
        print("## Mann Whitney U test on number of evaluations (evals).")
        print("p-values for alternative hypothesis row < column")
        print(display_test_grid(pvalues_evals), "\n")

        # append evals pvalues to aggregation df
        better_df = pd.DataFrame(
            [
                {
                    "model": stats.model,
                    "function": stats.function,
                    "alternative": "better",
                    "pvalue": row[0],
                }
                for row, (_, stats) in zip(
                    pvalues_evals[1:], list(stats_df.iterrows())[1:]
                )
            ]
        )
        worse_df = pd.DataFrame(
            [
                {
                    "model": stats.model,
                    "function": stats.function,
                    "alternative": "worse",
                    "pvalue": pvalue,
                }
                for pvalue, (_, stats) in zip(
                    pvalues_evals[0][1:], list(stats_df.iterrows())[1:]
                )
            ]
        )
        evals_pvalues_to_aggregate_df = pd.concat(
            [evals_pvalues_to_aggregate_df, better_df, worse_df], axis=0
        )

    # aggregate stats and save to csv
    aggregated_stats = aggregate_stats(stats_to_aggregate_df)

    print("# Aggregated stats")
    print(tabulate(aggregated_stats, headers="keys", tablefmt="github"), "\n")

    aggregated_stats.to_csv(
        "aggregated_stats.csv", index=False
    )

    # aggregate y pvalues and save to csv
    aggregated_y_pvalues = aggregate_pvalues(
        y_pvalues_to_aggregate_df, args.significance
    )

    print("# Aggregated y pvalues")
    print(
        tabulate(aggregated_y_pvalues, headers="keys", tablefmt="github"), "\n"
    )

    aggregated_y_pvalues.to_csv(
        "aggregated_y_pvalues.csv", index=False
    )

    # aggregate evals pvalues and save to csv
    aggregated_evals_pvalues = aggregate_pvalues(
        evals_pvalues_to_aggregate_df, args.significance
    )

    print("# Aggregated evals pvalues")
    print(
        tabulate(aggregated_evals_pvalues, headers="keys", tablefmt="github"),
        "\n",
    )

    aggregated_evals_pvalues.to_csv(
        "aggregated_evals_pvalues.csv", index=False
    )
