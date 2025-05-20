import pandas as pd
import numpy as np
import os

## @file data_generator.py
#  @brief Generates synthetic experimental data for conductive ink deposition experiments.
#         Outputs include process parameters and resulting electrical resistance.

## @brief Generates a synthetic dataset for conductive ink deposition.
#  Simulates multiple experimental parameters and a continuous resistance value.
#  Binary success is determined based on a threshold resistance.
#  @param n Number of samples to generate.
#  @return pd.DataFrame The generated dataset.
def make_data(n=100):
    # Experimental parameters (inputs)
    flow_rate = np.round(np.random.uniform(1.0, 10.0, n), 2)  # µL/s
    line_length = np.round(np.random.uniform(10, 100, n), 1)  # mm
    ink_concentration = np.round(np.random.uniform(30, 70, n), 1)  # wt%
    drying_time = np.round(np.random.uniform(30, 300, n), 0)  # seconds
    substrate_temp = np.round(np.random.uniform(20, 80, n), 1)  # °C
    nozzle_height = np.round(np.random.uniform(0.5, 2.0, n), 2)  # mm
    # Simulated resistance (Ω), influenced by inputs
    base_resistance = 5
    resistance = (
        base_resistance +
        (line_length / ink_concentration) * 2.5 +
        (100 - flow_rate) * 0.1 +
        (nozzle_height * 3) +
        np.random.normal(0, 2, n)  # noise
    )
    resistance = np.round(resistance, 2)

    # Define success as resistance below a threshold (e.g., 15 Ω)
    success_threshold = 20
    success = (resistance < success_threshold).astype(int)

    # Build DataFrame
    df = pd.DataFrame({
        'flow rate [µL/s]': flow_rate,
        'line length [mm]': line_length,
        'ink concentration [wt%]': ink_concentration,
        'drying time [sec]': drying_time,
        'substrate temperature [°C]': substrate_temp,
        'nozzle height [mm]': nozzle_height,
        'resistance [Ω]': resistance,
        'success [-]': success
    })

    return df

## @brief Appends data to CSV, creating file if it doesn't exist.
#  @param df DataFrame to save.
#  @param fname Output file name (default: "materials_data.csv").
def write_to_csv(df, fname="materials_data.csv"):
    if os.path.exists(fname):
        existing = pd.read_csv(fname)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(fname, index=False)