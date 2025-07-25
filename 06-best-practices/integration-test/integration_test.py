import os
import subprocess
from datetime import datetime

import pandas as pd

from s3io import save_data, read_data

BUCKET = os.getenv("BUCKET", "test-bucket")
INPUT_KEY = "data/2023-01.parquet"
OUTPUT_KEY = "predictions/2023-01.parquet"

def dt(h, m, s=0):
    return datetime(2023, 1, 1, h, m, s)

def make_df():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    cols = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    return pd.DataFrame(data, columns=cols)

def main():
    df = make_df()
    save_data(df, BUCKET, INPUT_KEY)

    # run batch.py
    subprocess.check_call(["python", "batch.py"])

    df_pred = read_data(BUCKET, OUTPUT_KEY)
    total = round(df_pred["predicted_duration"].sum(), 2)
    print("Sum:", total)
    # assert total == 36.28, f"Expected 36.28, got {total}"

if __name__ == "__main__":
    main()
