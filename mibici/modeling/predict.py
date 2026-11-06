"""Pronostico de demanda con un modelo del MLflow Model Registry.

El modelo se pide por alias (champion por defecto), de modo que promover otra
version en el registro cambia el modelo en uso sin tocar este codigo.

Uso:
    python -m mibici.modeling.predict                     # periodo de prueba con champion
    python -m mibici.modeling.predict --alias challenger
    python -m mibici.modeling.predict --start 2026-09-24
"""

from pathlib import Path
from typing import Annotated

from loguru import logger
import mlflow
from mlflow import MlflowClient
import pandas as pd
from sklearn.pipeline import Pipeline
import typer

from mibici.config import (
    CHAMPION_ALIAS,
    MLFLOW_TRACKING_URI,
    PANEL_FILE,
    PREDICTIONS_FILE,
    REGISTERED_MODEL_NAME,
    TEST_START,
)
from mibici.features import MODEL_FEATURES, TARGET
from mibici.modeling.evaluation import TIME_COLUMN, clip_counts, regression_metrics

app = typer.Typer()

FORECAST_COLUMN = "pronostico"


class DemandForecaster:
    """Carga una version del modelo registrado y genera el pronostico por estacion-hora."""

    def __init__(
        self,
        alias: str = CHAMPION_ALIAS,
        model_name: str = REGISTERED_MODEL_NAME,
        tracking_uri: str = MLFLOW_TRACKING_URI,
    ):
        mlflow.set_tracking_uri(tracking_uri)
        self.model_name = model_name
        self.alias = alias
        self.version = MlflowClient().get_model_version_by_alias(model_name, alias)
        self.model: Pipeline = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
        logger.info(
            f"Modelo {model_name} version {self.version.version} "
            f"({self.version.tags.get('modelo', 'sin etiqueta')}, alias {alias})"
        )

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Devuelve estacion, hora y pronostico. Si el panel trae el valor real, lo conserva."""
        keep = [c for c in ["station_id", TIME_COLUMN, TARGET] if c in panel.columns]
        out = panel[keep].copy()
        out[FORECAST_COLUMN] = clip_counts(self.model.predict(panel[MODEL_FEATURES]))
        out["model_version"] = int(self.version.version)
        return out


@app.command()
def main(
    panel_path: Path = PANEL_FILE,
    predictions_path: Path = PREDICTIONS_FILE,
    alias: str = CHAMPION_ALIAS,
    start: Annotated[str, typer.Option(help="Primera fecha a pronosticar.")] = TEST_START,
):
    if not Path(panel_path).exists():
        raise typer.BadParameter(f"No existe {panel_path}. Corre primero: make data")
    panel = pd.read_parquet(panel_path)
    panel = panel[panel[TIME_COLUMN] >= pd.Timestamp(start)]
    forecast = DemandForecaster(alias).predict(panel)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(predictions_path, index=False)
    if TARGET in forecast.columns:
        metrics = regression_metrics(forecast[TARGET], forecast[FORECAST_COLUMN])
        logger.info(
            "Error del pronostico: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        )
    logger.success(f"{len(forecast):,} pronosticos guardados en {predictions_path}")


if __name__ == "__main__":
    app()
