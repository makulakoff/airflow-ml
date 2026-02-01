from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowFailException

ARTIFACTS_DIR = Path("/opt/airflow/artifacts/ml_quickstart")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ARGS = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}

with DAG(
    dag_id="ml_quickstart_pipeline",
    description="Минимальный многоэтапный ML pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "quickstart"],
) as dag:

    @task
    def generate_data(n_samples: int = 500, seed: int = 42) -> dict:
        from sklearn.datasets import make_classification
        import numpy as np

        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=seed,
        )

        data_path = ARTIFACTS_DIR / "data.npz"
        np.savez(data_path, X=X, y=y)

        return {"data_path": str(data_path)}

    @task
    def preprocess(data_meta: dict) -> dict:
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        loaded = np.load(data_meta["data_path"])
        X, y = loaded["X"], loaded["y"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        out_path = ARTIFACTS_DIR / "preprocessed.npz"
        np.savez(
            out_path,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        return {"preprocessed_path": str(out_path)}

    @task
    def train_model(meta: dict) -> dict:
        import numpy as np
        import joblib
        from sklearn.linear_model import LogisticRegression

        data = np.load(meta["preprocessed_path"])
        model = LogisticRegression(max_iter=500)
        model.fit(data["X_train"], data["y_train"])

        model_path = ARTIFACTS_DIR / "model.joblib"
        joblib.dump(model, model_path)

        return {
            "model_path": str(model_path),
            "preprocessed_path": meta["preprocessed_path"],
        }

    @task
    def evaluate(meta: dict) -> dict:
        import numpy as np
        import joblib
        from sklearn.metrics import accuracy_score

        model = joblib.load(meta["model_path"])
        data = np.load(meta["preprocessed_path"])

        preds = model.predict(data["X_test"])
        acc = float(accuracy_score(data["y_test"], preds))

        metrics_path = ARTIFACTS_DIR / "metrics.json"
        metrics_path.write_text(
            json.dumps({"accuracy": acc}, indent=2),
            encoding="utf-8",
        )

        return {"accuracy": acc}

    @task
    def quality_gate(meta: dict):
        if meta["accuracy"] < 0.75:
            raise AirflowFailException(
                f"Accuracy too low: {meta['accuracy']:.3f}"
            )

    quality_gate(
        evaluate(
            train_model(
                preprocess(
                    generate_data()
                )
            )
        )
    )