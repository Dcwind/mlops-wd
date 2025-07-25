import os
import sys
import pickle
import pandas as pd

from s3io import read_data, save_data

CATEGORICAL = ['PULocationID', 'DOLocationID']

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype(int).astype(str)
    return df

def run(year: int, month: int, bucket: str, input_key: str, output_key: str):
    with open('../model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    df = read_data(bucket, input_key)
    df = prepare_data(df)

    X = dv.transform(df[CATEGORICAL].to_dict(orient='records'))
    preds = model.predict(X)

    df_result = pd.DataFrame({
        "ride_id": f"{year:04d}/{month:02d}_" + df.index.astype(str),
        "predicted_duration": preds
    })

    save_data(df_result, bucket, output_key)

if __name__ == "__main__":
    # hard coded, keep it minimal
    YEAR = 2023
    MONTH = 1
    BUCKET = os.getenv("BUCKET", "test-bucket")        
    INPUT_KEY = "data/2023-01.parquet"
    OUTPUT_KEY = "predictions/2023-01.parquet"

    run(YEAR, MONTH, BUCKET, INPUT_KEY, OUTPUT_KEY)
