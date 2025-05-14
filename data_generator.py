import pandas as pd
import numpy as np
import os

## @file data_generator.py
#  @brief Generates synthetic experimental data for materials synthesis.
#         Data includes process parameters and a binary success label.

## @brief Generates a synthetic dataset of materials synthesis experiments.
#  Includes randomized parameters and probabilistic success labels.
#  @param n Number of samples to generate.
#  @return pd.DataFrame The generated dataset.
def make_data(n=100):
    temp = np.random.uniform(80, 200, n)
    speed = np.random.uniform(1000, 4000, n)
    ratio = np.random.uniform(0.2, 0.8, n)
    time = np.random.uniform(10, 60, n)
    concentration = np.random.uniform(0.1, 1.0, n)
    anneal_time = np.random.uniform(30, 300, n)

    prob_success = (
        (temp > 120).astype(int) +
        ((1500 < speed) & (speed < 3500)).astype(int) +
        ((0.3 < ratio) & (ratio < 0.7)).astype(int) +
        ((0.2 < concentration) & (concentration < 0.8)).astype(int) +
        ((60 < anneal_time) & (anneal_time < 240)).astype(int)
    ) / 5.0

    success = (np.random.rand(n) < prob_success).astype(int)

    df = pd.DataFrame({
        'temperature': temp,
        'spin_speed': speed,
        'solvent_ratio': ratio,
        'deposition_time': time,
        'solute_concentration': concentration,
        'annealing_time': anneal_time,
        'success': success
    })

    return df

## @brief Saves a DataFrame to CSV, appending to an existing file if present.
#  @param df DataFrame to save.
#  @param fname Output file name (default: "materials_data.csv").
def write_to_csv(df, fname="materials_data.csv"):
    if os.path.exists(fname):
        existing = pd.read_csv(fname)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(fname, index=False)
