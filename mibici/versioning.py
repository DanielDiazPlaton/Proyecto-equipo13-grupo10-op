"""Versionado de datasets con MLflow: cada version queda como un run con su huella."""

import hashlib
import json
from pathlib import Path

import mlflow
import pandas as pd

from mibici.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Hash SHA-256 del contenido. Si un dato cambia, cambia la huella."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


class DatasetVersioner:
    """Registra versiones de datos en MLflow (parametros, metricas, esquema y archivo)."""

    def __init__(
        self, tracking_uri: str = MLFLOW_TRACKING_URI, experiment: str = MLFLOW_EXPERIMENT
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    def log_version(
        self,
        df: pd.DataFrame,
        name: str,
        version: str,
        stage: str,
        params: dict | None = None,
        metrics: dict | None = None,
        artifacts: list[Path] | None = None,
        file_path: Path | None = None,
        parent_version: str | None = None,
    ) -> str:
        fingerprint = dataframe_fingerprint(df)
        with mlflow.start_run(run_name=f"{name}-{version}") as run:
            mlflow.set_tags(
                {
                    "data.name": name,
                    "data.version": version,
                    "data.stage": stage,
                    "data.sha256": fingerprint,
                    "data.parent_version": parent_version or "ninguna",
                }
            )
            source = str(file_path) if file_path else name
            dataset = mlflow.data.from_pandas(df.head(1000), source=source, name=name)
            mlflow.log_input(dataset, context=stage)
            mlflow.log_params(params or {})
            base = {
                "n_rows": len(df),
                "n_cols": df.shape[1],
                "pct_nulls": float(df.isna().mean().mean() * 100),
            }
            mlflow.log_metrics({**base, **(metrics or {})})
            schema = {c: str(t) for c, t in df.dtypes.items()}
            mlflow.log_text(json.dumps(schema, indent=2), "schema.json")
            for path in artifacts or []:
                mlflow.log_artifact(str(path))
            if file_path is not None:
                mlflow.log_artifact(str(file_path), artifact_path="data")
            return run.info.run_id

    @staticmethod
    def history() -> pd.DataFrame:
        runs = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT], order_by=["attributes.start_time ASC"]
        )
        cols = [
            "tags.data.version",
            "tags.data.stage",
            "tags.data.parent_version",
            "metrics.n_rows",
            "metrics.n_cols",
            "tags.data.sha256",
            "run_id",
        ]
        return runs[[c for c in cols if c in runs.columns]]
