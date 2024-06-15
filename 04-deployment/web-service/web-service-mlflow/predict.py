import pickle
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify


RUN_ID = "325dfd907f904e358ecd882849cb26d5"
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)

path = client.download_artifacts(run_id = RUN_ID, path= "./artifacts")
print(f"downloading the dict vectorizer to {path}")

with open(path, 'rb') as f_out:
    dv = pickle.load(f_out)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'],ride["DOLocationID"])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    x = dv.transform(features)
    preds = model.predict(x)
    return float(preds[0])

app = Flask('duration-prediction')

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred 
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host= '0.0.0.0', port=9694)

    

    
