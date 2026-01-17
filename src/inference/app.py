from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc

app = FastAPI()

MODEL_NAME = "fraudflow_model"
MODEL_STAGE = "None"  

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    preds = model.predict(df)
    return {"fraud_probability": float(preds[0])}

'''
irm http://localhost:8000/predict `
-Method Post `
-ContentType 'application/json' `
-Body '{
"Time": 0,
"V1": -1.3598,
"V2": -0.0727,
"V3": 2.5363,
"V4": 1.3781,
"V5": -0.3383,
"V6": 0.4623,
"V7": 0.2395,
"V8": 0.0987,
"V9": 0.3638,
"V10": 0.0908,
"V11": -0.5516,
"V12": -0.6178,
"V13": -0.9913,
"V14": -0.3112,
"V15": 1.4682,
"V16": -0.4704,
"V17": 0.2079,
"V18": 0.0257,
"V19": 0.4039,
"V20": 0.2514,
"V21": -0.0183,
"V22": 0.2778,
"V23": -0.1105,
"V24": 0.0669,
"V25": 0.1285,
"V26": -0.1891,
"V27": 0.1335,
"V28": -0.0210,
"Amount": 149.62
}'

'''
