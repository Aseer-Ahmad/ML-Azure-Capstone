# score.py
import joblib
import json
import os
# from azureml.core.model import Model

def init():
    global model
    # model_path = Model.get_model_path('automl_best_model:4')
    # model_path = os.path.join(model_path, "model.pkl")
    model_path = os.path.join("outputs", "model.pkl")
    model = joblib.load(model_path)

def run(data):
    try:
        input_data = json.loads(data)
        result = model.predict([input_data])
        return result.tolist()
    except Exception as e:
        return str(e)