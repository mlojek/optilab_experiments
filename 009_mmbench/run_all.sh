#!/bin/bash
set -e

cd "$(dirname "$0")"

NUM_SAMPLES=51

python create_dataset.py 10 --num_samples $NUM_SAMPLES
python create_dataset.py 30 --num_samples $NUM_SAMPLES
python test_interpolation.py --dataset dataset_10d.json --output interpolation_10d.csv
python test_interpolation.py --dataset dataset_30d.json --output interpolation_30d.csv
python test_extrapolation.py --dataset dataset_10d.json --output extrapolation_10d.csv
python test_extrapolation.py --dataset dataset_30d.json --output extrapolation_30d.csv
