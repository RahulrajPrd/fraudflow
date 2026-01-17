import pandas as pd
import requests
import time
import random
import os

DATA_PATH = "data/processed/test.csv"
API_URL = "http://localhost:8000/predict"
OUTPUT_PATH = "data/stream/predictions.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    os.makedirs("data/stream", exist_ok=True)

    # Create output file with header if not exists
    if not os.path.exists(OUTPUT_PATH):
        df.head(0).assign(prediction=0.0).to_csv(OUTPUT_PATH, index=False)

    while True:
        # Pick random transaction
        row = df.sample(1)

        # Extract ground truth
        y_true = int(row["Class"].values[0])

        # Remove label before sending
        payload = row.drop("Class", axis=1).to_dict(orient="records")[0]

        # Call API
        response = requests.post(API_URL, json=payload)
        y_pred = response.json()["fraud_probability"]

        # Store result
        record = row.copy()
        record["prediction"] = y_pred
        record.to_csv(OUTPUT_PATH, mode="a", header=False, index=False)

        print(f"Sent transaction | True: {y_true} | Pred: {y_pred}")

        time.sleep(random.uniform(0.5, 2.0))  # simulate random arrival

if __name__ == "__main__":
    main()
