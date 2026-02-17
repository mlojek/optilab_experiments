#!/usr/bin/env python
"""
Run the full benchmark pipeline: create datasets and run interpolation/extrapolation tests.
"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

steps = [
    ("Creating dataset dim=30", [sys.executable, "create_dataset.py", "30", "--num_samples", "10"]),
    ("Running interpolation dim=10", [sys.executable, "test_interpolation.py", "--dataset", "dataset_10d.json", "--output", "interpolation_10d.csv"]),
    ("Running interpolation dim=30", [sys.executable, "test_interpolation.py", "--dataset", "dataset_30d.json", "--output", "interpolation_30d.csv"]),
    ("Running extrapolation dim=10", [sys.executable, "test_extrapolation.py", "--dataset", "dataset_10d.json", "--output", "extrapolation_10d.csv"]),
    ("Running extrapolation dim=30", [sys.executable, "test_extrapolation.py", "--dataset", "dataset_30d.json", "--output", "extrapolation_30d.csv"]),
]

# Skip dataset creation if files already exist
if os.path.exists("dataset_10d.json"):
    print("dataset_10d.json already exists, skipping.")
else:
    steps.insert(0, ("Creating dataset dim=10", [sys.executable, "create_dataset.py", "10", "--num_samples", "10"]))

if os.path.exists("dataset_30d.json"):
    print("dataset_30d.json already exists, skipping dim=30 creation.")
    steps = [s for s in steps if "dim=30" not in s[0] or "Creating" not in s[0]]

for desc, cmd in steps:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: {desc}")
        sys.exit(1)

print("\n\nAll steps completed successfully!")
