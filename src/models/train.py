import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import yaml
from sklearn.metrics import roc_auc_score
import os

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()["train"]

    train_df = pd.read_csv(params["test_path"])
    val_df = pd.read_csv(params["val_path"])

    X_train = train_df.drop("Class", axis=1)
    y_train = train_df["Class"]

    X_val = val_df.drop("Class", axis=1)
    y_val = val_df["Class"]

    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        eval_metric="auc"
    )

    mlflow.set_experiment("fraudflow")

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)

        mlflow.log_params({
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"]
        })

        mlflow.log_metric("val_auc", auc)

        # Log to MLflow registry (control plane)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="fraudflow_model"
        )

        # Save production-ready serialized model (runtime plane)
        os.makedirs("models/latest_model", exist_ok=True)
        mlflow.xgboost.save_model(model, path="models/latest_model")

        print("Validation AUC:", auc)
        print("Serialized model saved to models/latest_model")

if __name__ == "__main__":
    main()
