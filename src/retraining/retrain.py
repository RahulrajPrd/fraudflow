import subprocess

def retrain():
    print("Drift detected → Retraining triggered")

    # Call original training script
    subprocess.run(["python", "src/models/train.py"], check=True)

    print("New model trained and saved to models/latest_model")

if __name__ == "__main__":
    retrain()
