````md
# Apache Airflow

Документ предназначен для быстрого понимания **Apache Airflow** и запуска **многоэтапного ML-пайплайна** на Python локально.

---

## 1. Что такое Apache Airflow

**Apache Airflow** — это оркестратор задач. Он управляет:
- **порядком выполнения**
- **расписанием**
- **повторами и ошибками**
- **логами и мониторингом**

Airflow **не предназначен** для тяжёлых вычислений сам по себе — он управляет запуском вашего кода.

---

## 2. Основные сущности

### DAG (Directed Acyclic Graph)
Граф задач без циклов.  
Описывает pipeline целиком.

### Task
Одна задача внутри DAG (узел графа).

### Operator
Шаблон задачи:
- `PythonOperator`
- `BashOperator`
- `DockerOperator`
- и др.

### Scheduler
Решает, **когда** запускать DAG и задачи.

### Executor
Решает, **как** выполнять задачи:
- `SequentialExecutor`
- `LocalExecutor`
- `CeleryExecutor`
- `KubernetesExecutor`

### Web UI
Интерфейс для:
- запуска DAG
- просмотра логов
- перезапуска задач
- анализа ошибок

---

## 3. Ключевые принципы Airflow

1. **DAG — это описание, а не выполнение**
   - Нельзя делать тяжёлые вычисления при импорте DAG-файла.

2. **Идемпотентность**
   - Любая задача должна безопасно перезапускаться.

3. **XCom — только для метаданных**
   - Большие данные передаются через файлы / хранилища.

4. **Наблюдаемость**
   - Логи, артефакты, retries важнее “красивого” кода.

---

## 4. Минимальный запуск Airflow через Docker

### Структура проекта

```text
airflow-ml/
├── dags/
│   └── ml_quickstart_dag.py
├── artifacts/
├── docker-compose.yaml
└── .env
````

---

### docker-compose.yaml

```yaml
version: "3.8"

x-airflow-common: &airflow-common
  image: apache/airflow:2.9.3
  environment:
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: sqlite:////opt/airflow/airflow.db
    AIRFLOW__CORE__LOAD_EXAMPLES: "false"
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: "false"
  volumes:
    - ./dags:/opt/airflow/dags
    - ./artifacts:/opt/airflow/artifacts
  user: "${AIRFLOW_UID:-50000}:0"

services:
  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - "8080:8080"

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler

  airflow-init:
    <<: *airflow-common
    command: >
      bash -c "
      airflow db init &&
      airflow users create
        --username admin
        --password admin
        --firstname Admin
        --lastname User
        --role Admin
        --email admin@example.com
      "
```

---

### .env (рекомендуется)

```bash
AIRFLOW_UID=$(id -u)
```

---

### Запуск

```bash
docker compose up airflow-init
docker compose up
```

Web UI: [http://localhost:8080](http://localhost:8080)
Логин / пароль: `admin / admin`

---

## 5. Пример ML DAG (несколько этапов)

Pipeline:

1. Генерация данных
2. Препроцессинг
3. Обучение модели
4. Оценка качества
5. Проверка порога качества

---

### ml_quickstart_dag.py

```python
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
```

---

## 6. Что проверить после запуска

* Статус DAG и задач в UI
* Логи каждой задачи
* Артефакты в папке:

```text
artifacts/ml_quickstart/
├── data.npz
├── preprocessed.npz
├── model.joblib
└── metrics.json
```

