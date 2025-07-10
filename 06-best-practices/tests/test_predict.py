import pandas as pd
from datetime import datetime

import predict

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    # Input: no duration
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    # Expected (only valid duration & no missing IDs)
    expected_data = [
        (None, None, dt(1, 1), dt(1, 10), 9.0),
        (1, 1, dt(1, 2), dt(1, 10), 8.0),
    ]

    expected_columns = columns + ["duration"]
    expected_df = pd.DataFrame(expected_data, columns=expected_columns).reset_index(drop=True)

    actual_df = predict.prepare_data(df).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual_df, expected_df)

