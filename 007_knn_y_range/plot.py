'''
Script to plot the correlation between knn Y range and pvalues against ipop.
'''
import argparse
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import math


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pvalues_csv", type=Path, help="Path to CSV file with pvalues.",)
    parser.add_argument("y_range_csv", type=Path, help="Path to CSV file with KNN Y range results",)
    args = parser.parse_args()

    pvalues_df = pd.read_csv(args.pvalues_csv)
    if 'alternative' in pvalues_df.columns:
        pvalues_df = pvalues_df[pvalues_df['alternative'] == 'better'].copy()
        pvalues_df.drop(columns=['alternative'], inplace=True)

    # remove rows where function == 'total' (case-insensitive, ignoring surrounding whitespace)
    if 'function' in pvalues_df.columns:
        mask = ~pvalues_df['function'].astype(str).str.strip().str.lower().eq('total')
        pvalues_df = pvalues_df[mask].copy()

    y_range_df = pd.read_csv(args.y_range_csv)

    # drop first row and first column from both dataframes
    pvals_arr = pvalues_df.iloc[:, 1:].to_numpy()
    yrange_arr = y_range_df.iloc[:, 1:].to_numpy()

    # ensure they have the same shape
    print(pvals_arr.shape)
    assert pvals_arr.shape == yrange_arr.shape, f"Shapes differ: {pvals_arr.shape} vs {yrange_arr.shape}"

    # pair corresponding elements into a list of (pvalue, y_range) tuples
    pairs = [(float(a), float(b)) for a, b in zip(pvals_arr.ravel(), yrange_arr.ravel())]

    # filter out non-finite values
    pairs = [(p, y) for p, y in pairs if math.isfinite(p) and math.isfinite(y)]

    # unpack
    pvals = [p for p, _ in pairs]
    yranges = [y for _, y in pairs]

    # plot
    plt.figure(figsize=(8, 6))
    plt.scatter(pvals, yranges, s=20, alpha=0.7)
    plt.xlabel("pvalue")
    plt.ylabel("KNN Y range")
    plt.title("pvalue vs KNN Y range")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
