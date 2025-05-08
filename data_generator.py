import pandas as pd
import numpy as np
import os


# data_generator.py
# Creates synthetic materials synthesis data and saves it to a CSV file.

def make_data(n=10):
    """
    Generates a batch of synthetic experiment data.

    Each row includes temperature, spin speed, solvent ratio, and deposition time,
    plus a binary success label based on some simple rules.

    Parameters:
        n (int): Number of data points to generate.

    Returns:
        pd.DataFrame: The generated dataset.
    """
    temp = np.random.uniform(80, 200, n)
    speed = np.random.uniform(1000, 4000, n)
    ratio = np.random.uniform(0.2, 0.8, n)
    time = np.random.uniform(10, 60, n)

    # Basic logic for success: tuned to favor moderate process conditions
    success = (
            (temp > 120) &
            (speed > 1500) & (speed < 3500) &
            (ratio > 0.3) & (ratio < 0.7)
    ).astype(int)

    df = pd.DataFrame({
        'temperature': temp,
        'spin_speed': speed,
        'solvent_ratio': ratio,
        'deposition_time': time,
        'success': success })
    return df


def write_to_csv(df, fname="materials_data.csv"):
    """
    Appends new data to a CSV file (or creates it if it doesn't exist).

    Parameters:
        df (pd.DataFrame): New data to save.
        fname (str): File name for the dataset.
    """
    if os.path.exists(fname):
        existing = pd.read_csv(fname)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(fname, index=False)


if __name__ == "__main__":
    d = make_data(20)
    write_to_csv(d)
