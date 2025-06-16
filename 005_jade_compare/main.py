'''
Compare KNN-CMA-ES and KNN-JADE with statistical tests.
JADE results are simulated based on reported median and iqr values.
'''
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate

from optilab.utils.pickle_utils import list_all_pickles, load_from_pickle
from optilab.utils.stat_test import mann_whitney_u_test_grid
from optilab.utils.aggregate_pvalues import aggregate_pvalues


TOLERANCE = 1e-8


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmaes_dir', type=Path, help='Path to directory with CMA-ES results pickles.',)
    parser.add_argument('jade_file', type=Path, help='Path to CSV file with JADE results.',)
    parser.add_argument('--significance', type=float, default=0.05, help='Significance value for stat tests.',)
    args = parser.parse_args()

    cmaes_values = {}
    jade_values = {}

    # READ CMAES
    for file_path in tqdm(list_all_pickles(args.cmaes_dir), desc='Reading CMAES'):
        data = load_from_pickle(file_path)
        cmaes_values[data[0].function_metadata.name] = [run.bests_y() for run in data]

    print(cmaes_values.keys())

    # READ JADE
    df = pd.read_csv(args.jade_file)

    methods = df.columns[2:]

    for i in range(0, len(df), 2):
        func_name = df.iloc[i, 0]
        medians = df.iloc[i, 2:].values.astype(float)
        iqrs = df.iloc[i+1, 2:].values.astype(float)
        
        simulated_values = []
        for median, iqr in zip(medians, iqrs):
            if iqr == 0:
                values = [median] * 51
            else:
                scale = iqr / 1.348
                values = np.random.laplace(loc=median, scale=scale, size=51).tolist()
            values = [max(v, TOLERANCE) for v in values]
            simulated_values.append(values)
        
        jade_values[func_name] = simulated_values

    print(jade_values.keys())

    # PVALUES AGGREGATION
    assert set(cmaes_values.keys()) == set(jade_values.keys())

    pvalues_aggregation_df = pd.DataFrame(
        columns=["model", "function", "alternative", "pvalue"]
    )

    for function_name in cmaes_values.keys():
        for cma_vals, jade_vals, bufsize in zip(cmaes_values[function_name], jade_values[function_name], ['0', '2', '5', '10', '20', '30', '50']):
            test_grid = mann_whitney_u_test_grid([cma_vals, jade_vals])

            pvalues_df = pd.DataFrame(
                [
                    {
                        "model": bufsize,
                        "function": function_name,
                        "alternative": "better",
                        "pvalue": test_grid[0][1],
                    },
                    {
                        "model": bufsize,
                        "function": function_name,
                        "alternative": "worse",
                        "pvalue": test_grid[1][0],
                    }
                ]
            )
            pvalues_aggregation_df = pd.concat(
                [pvalues_aggregation_df, pvalues_df], axis=0
            )

    print(tabulate(aggregate_pvalues(pvalues_aggregation_df, args.significance), headers='keys', tablefmt='github',))

    pvalues_aggregation_df.to_csv(
        "aggregated_pvalues.csv", index=False
    )

