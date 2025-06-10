#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

#https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet

def read_dataframe(year, month, taxi):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi}_tripdata_{year}-{month:02d}.parquet'
    print(f'Data url: {url}')
    df = pd.read_parquet(url)
    print(f'Data shape: {df.shape}')

    if taxi=='green':
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    else:
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime

    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    print(f'Data shape after filtering: {df.shape}')

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None):
    # categorical = ['PU_DO']
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        mlflow.set_tag("developer", "Daniel")
        mlflow.set_tag("model_type", "Linear Regression")
        mlflow.set_tag("param", "defualt")
        
        model = LinearRegression()
        model.fit(X_train, y_train)

        print("Model intercept:", model.intercept_)
        # print("Model coefficients:", model.coef_)

        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")

        return run.info.run_id


def run(year, month, taxi):
    df_train = read_dataframe(year=year, month=month, taxi=taxi)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month, taxi=taxi)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    parser.add_argument('--taxi', required=True, help='The colour of the taxi, green or yellow')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month, taxi=args.taxi)

    with open("run_id.txt", "w") as f:
        f.write(run_id)