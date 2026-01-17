import sys
from pathlib import Path

# Add project root to path so imports work
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.drift_check import run_drift_check
from src.retraining.retrain import retrain

def main():
    drift = run_drift_check()
    if drift:
        retrain()
    else:
        print("No drift detected → No retraining required")

if __name__ == "__main__":
    main()