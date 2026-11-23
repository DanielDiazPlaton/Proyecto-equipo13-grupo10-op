"""Exportacion del modelo champion a una carpeta lista para servirse (Fase 3).

El servicio y la imagen Docker no deben depender de mlflow.db. Este modulo toma una
version del Model Registry por alias y deja en models/mibici-demanda-salidas/:

- model/: el pipeline en formato MLflow (MLmodel y el archivo skops).
- estaciones.csv: los atributos fijos de cada estacion, un renglon por estacion.
- model_info.json: nombre, version, alias, model_uri, run, hiperparametros, metricas,
  version de los datos y versiones de las librerias.

Uso:
    python -m mibici.modeling.export                     # champion
    python -m mibici.modeling.export --alias challenger
"""

from datetime import datetime
import json
from pathlib import Path
import platform
import shutil

from loguru import logger
import mlflow
from mlflow import MlflowClient
import pandas as pd
import sklearn
import typer

from mibici.config import (
    CHAMPION_ALIAS,
    MLFLOW_TRACKING_URI,
    PANEL_FILE,
    REGISTERED_MODEL_NAME,
    SERVING_INFO_FILE,
    SERVING_MODEL_DIR,
    SERVING_MODEL_SUBDIR,
    SERVING_STATIONS_FILE,
    TEST_START,
)
from mibici.features import MODEL_FEATURES

app = typer.Typer()

# Atributos que no cambian con la hora: la API los toma de estaciones.csv.
STATION_ATTRIBUTES = [
    "station_id",
    "lat",
    "lon",
    "capacity",
    "zone",
    "pc1",
    "pc2",
    "pc3",
    "station_cluster",
]
VALIDATION_METRICS = ["cv_rmse", "cv_mae"]
TEST_METRICS = ["MAE", "RMSE", "WAPE_pct", "sesgo"]


def station_attributes(panel: pd.DataFrame) -> pd.DataFrame:
    """Un renglon por estacion con sus atributos fijos, tomados del panel."""
    stations = panel[STATION_ATTRIBUTES].drop_duplicates()
    stations = stations.sort_values("station_id").reset_index(drop=True)
    repeated = stations["station_id"].duplicated()
    if repeated.any():
        ids = stations.loc[repeated, "station_id"].tolist()
        raise ValueError(f"Estaciones con mas de un valor en sus atributos fijos: {ids[:10]}")
    return stations


class ModelExporter:
    """Copia una version del Model Registry, con su metadata, a una carpeta local."""

    def __init__(
        self,
        alias: str = CHAMPION_ALIAS,
        model_name: str = REGISTERED_MODEL_NAME,
        tracking_uri: str = MLFLOW_TRACKING_URI,
    ):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = model_name
        self.alias = alias
        self.version = self.client.get_model_version_by_alias(model_name, alias)
        self.model_uri = f"models:/{model_name}/{self.version.version}"
        logger.info(f"Modelo a exportar: {self.model_uri} (alias {alias})")

    def model_info(self, model_dir: Path, n_stations: int) -> dict:
        """Metadata del modelo exportado, tomada del registro y del run que lo entreno."""
        tags = self.version.tags
        run = self.client.get_run(self.version.run_id)
        metrics = run.data.metrics

        validation = {}
        for name in VALIDATION_METRICS:
            validation[name] = metrics[name]
        test = {}
        for name in TEST_METRICS:
            test[name] = metrics[f"test_{name}"]

        # Se carga la copia exportada: confirma que el modelo se puede leer sin el registro.
        pipeline = mlflow.sklearn.load_model(str(model_dir))
        estimator = pipeline.named_steps["model"]

        return {
            "registered_model_name": self.model_name,
            "version": int(self.version.version),
            "alias": self.alias,
            "model_uri": self.model_uri,
            "model_uri_alias": f"models:/{self.model_name}@{self.alias}",
            "run_id": self.version.run_id,
            "algoritmo": tags.get("modelo", ""),
            "estimador": f"{type(estimator).__module__}.{type(estimator).__name__}",
            "hiperparametros": json.loads(tags.get("hiperparametros", "{}")),
            "metricas_validacion": validation,
            "metricas_prueba": test,
            "periodo_prueba_desde": TEST_START,
            "variables": MODEL_FEATURES,
            "estaciones": n_stations,
            "data.version": tags.get("data.version", ""),
            "data.sha256": tags.get("data.sha256", ""),
            "fecha_exportacion": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "mlflow": mlflow.__version__,
        }

    def export(self, panel: pd.DataFrame, output_dir: Path = SERVING_MODEL_DIR) -> dict:
        """Escribe model/, estaciones.csv y model_info.json en output_dir."""
        output_dir = Path(output_dir)
        model_dir = output_dir / SERVING_MODEL_SUBDIR
        # Se borra la copia anterior para no mezclar archivos de dos versiones.
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True)

        logger.info(f"Descargando {self.model_uri}...")
        mlflow.artifacts.download_artifacts(artifact_uri=self.model_uri, dst_path=str(model_dir))

        stations = station_attributes(panel)
        stations.to_csv(output_dir / SERVING_STATIONS_FILE, index=False)
        logger.info(f"Atributos de {len(stations)} estaciones guardados")

        info = self.model_info(model_dir, len(stations))
        info_text = json.dumps(info, indent=2, ensure_ascii=False)
        (output_dir / SERVING_INFO_FILE).write_text(info_text + "\n", encoding="utf-8")
        logger.success(
            f"Modelo listo: {self.model_uri} ({info['algoritmo']}) exportado en {output_dir}"
        )
        return info


@app.command()
def main(
    panel_path: Path = PANEL_FILE,
    output_dir: Path = SERVING_MODEL_DIR,
    alias: str = CHAMPION_ALIAS,
):
    if not Path(panel_path).exists():
        raise typer.BadParameter(f"No existe {panel_path}. Corre primero: make data")
    panel = pd.read_parquet(panel_path)
    ModelExporter(alias).export(panel, output_dir)


if __name__ == "__main__":
    app()
