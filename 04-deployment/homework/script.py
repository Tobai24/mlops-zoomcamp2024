
import pickle
import pandas as pd


with open("model.bin", 'rb') as f:
    dv, model = pickle.load(f)

categorical = ["PULocationID", "DOLocationID"]


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-05.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)
    
print(f"Mean of predictions for May: {round(y_pred.mean(), 3)}")
 