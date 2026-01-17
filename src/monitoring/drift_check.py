import pandas as pd
import os
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# --- Configuration ---
REFERENCE_DATA_PATH = "data/processed/train.csv"
CURRENT_DATA_PATH = "data/stream/predictions.csv"
REPORT_DIR = "reports"
REPORT_JSON_PATH = os.path.join(REPORT_DIR, "drift_report.json")

def run_drift_check():
    # Ensure report directory exists
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    if not os.path.exists(REFERENCE_DATA_PATH) or not os.path.exists(CURRENT_DATA_PATH):
        print(f"Error: Files not found. Check paths.")
        return

    ref_df = pd.read_csv(REFERENCE_DATA_PATH)
    curr_df = pd.read_csv(CURRENT_DATA_PATH)

    # Drop non-feature columns safely
    ref_df = ref_df.drop(columns=[c for c in ["Class"] if c in ref_df.columns])
    curr_df = curr_df.drop(columns=[c for c in ["Class", "prediction"] if c in curr_df.columns])

    # Run drift report
    print("Running drift calculation...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df)

    # Save JSON only
    print(f"Saving JSON report to {REPORT_JSON_PATH}...")
    report.save_json(REPORT_JSON_PATH)

    # (Optional) Read drift flag for quick CLI feedback
    with open(REPORT_JSON_PATH) as f:
        report_json = json.load(f)

    drift_detected = report_json["metrics"][0]["result"]["dataset_drift"]
    print("Drift detected:", drift_detected)

    print("Drift check complete.")

if __name__ == "__main__":
    run_drift_check()
