# s3io.py
import os
from io import BytesIO

import boto3
import pandas as pd

def _s3():
    return boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL"))

def save_data(df: pd.DataFrame, bucket: str, key: str):
    buf = BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression=None, index=False)
    buf.seek(0)
    _s3().upload_fileobj(buf, bucket, key)

def read_data(bucket: str, key: str) -> pd.DataFrame:
    obj = _s3().get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()))
