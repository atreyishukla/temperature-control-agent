"""
Seed logs/experience.csv with rows from the historical Excel file.

Run this to simulate accumulated sensor data so you can test retrain.py
without waiting for real readings to come in.

Usage:
    python seed_experience.py            # seeds 48 rows (2 days)
    python seed_experience.py --rows 200
"""

import argparse
import csv
import os

import pandas as pd
from data_pipeline import load_data

LOG_PATH  = 'logs/experience.csv'
LOG_FIELDS = ['T_outside', 'T_inside', 'T_floor', 'SR_direct', 'fan_on', 'heater_on']


def seed(n_rows: int = 48, data_path: str = 'data/Concrete_floor_results.xlsx') -> None:
    df = load_data(data_path)

    # Use the validation slice (rows 6132-7316) so we don't overlap training data
    df = df.iloc[6132: 6132 + n_rows].copy()
    df['fan_on']    = 0
    df['heater_on'] = 0

    os.makedirs(os.path.dirname(LOG_PATH) or '.', exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        for _, row in df.iterrows():
            writer.writerow({k: row[k] for k in LOG_FIELDS})

    print(f'Seeded {len(df)} rows → {LOG_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=48)
    args = parser.parse_args()
    seed(args.rows)
