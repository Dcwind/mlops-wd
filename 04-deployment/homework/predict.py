#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd
import numpy as np
import boto3
from io import BytesIO

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def save_predictions(df, preds, year, month, output_file):
    # create ride_id
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype(str)

    # assemble results
    df_result = pd.DataFrame({
        "ride_id": df["ride_id"],
        "predicted_duration": preds
    })

    # # write to Parquet
    # df_result.to_parquet(
    #     output_file,
    #     engine="pyarrow",
    #     compression=None,
    #     index=False
    # )

    # write Parquet to an in‑memory buffer
    buffer = BytesIO()
    df_result.to_parquet(buffer, engine="pyarrow", compression=None, index=False)
    buffer.seek(0)

    # build S3 key and upload
    bucket = "mlops-s3-bucket-test"
    prefix = "taxi/predictions"
    key = f"{prefix.rstrip('/')}/{output_file}"
    s3 = boto3.client("s3")
    s3.upload_fileobj(buffer, bucket, key)

    return f"s3://{bucket}/{key}"

def run(year: int, month: int) -> str:
    """
    Load the model, read data for the given year/month,
    run predictions, save results, and return output path.
    """
    # load vectorizer and model
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # data URL
    url = (
        f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    )

    # predict
    df = read_data(url)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    preds = model.predict(X_val)

    print(f'Predictions shape: {preds.shape}')
    print(f'Mean prediction: {np.mean(preds):.2f} minutes')

    # save
    output_file = (
        f'yellow_tripdata_{year:04d}-{month:02d}_predictions.parquet'
    )
    output_location = save_predictions(df, preds, year, month, output_file)
    print(f'Saved predictions to {output_location}')

    return output_file

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run duration prediction for a specific year and month')
    parser.add_argument('--year', type=int, required=True, help='Year (YYYY) of the data to use for inference')
    parser.add_argument('--month', type=int, required=True, help='Month (1-12) of the data to use for inference')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.year, args.month)

