import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# --- Configuration ---
REFERENCE_DATA_PATH = "data/processed/train.csv"
CURRENT_DATA_PATH = "data/stream/predictions.csv"
REPORT_DIR = "reports"
REPORT_PATH = os.path.join(REPORT_DIR, "drift_report.html")

def main():
    # 1. Ensure report directory exists
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 2. Load Data
    print("Loading data...")
    if not os.path.exists(REFERENCE_DATA_PATH) or not os.path.exists(CURRENT_DATA_PATH):
        print(f"Error: Files not found. Check {REFERENCE_DATA_PATH} and {CURRENT_DATA_PATH}")
        return
    
    ref_df = pd.read_csv(REFERENCE_DATA_PATH)
    curr_df = pd.read_csv(CURRENT_DATA_PATH)

    # 3. Preprocessing: Remove target/prediction for feature drift
    # Drop "Class" and "prediction" only if they exist
    ref_df = ref_df.drop(columns=[c for c in ["Class"] if c in ref_df.columns])
    curr_df = curr_df.drop(columns=[c for c in ["Class", "prediction"] if c in curr_df.columns])

    # 4. Run Report
    print("Running drift calculation...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df)

    # 5. Save Report
    print(f"Saving report to {REPORT_PATH}...")
    report.save_html(REPORT_PATH)
    print("Drift check complete.")

if __name__ == "__main__":
    main()