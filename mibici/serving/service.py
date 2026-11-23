"""Servicio de pronostico: del cuerpo de la API al pronostico del modelo exportado.

Lee la carpeta que deja mibici.modeling.export (model/, estaciones.csv y model_info.json),
de modo que no necesita mlflow.db ni el Model Registry.
"""

import json
from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from mibici.config import SERVING_INFO_FILE, SERVING_MODEL_SUBDIR, SERVING_STATIONS_FILE
from mibici.features import MODEL_FEATURES, DemandFeatureBuilder
from mibici.modeling.evaluation import clip_counts
from mibici.modeling.export import STATION_ATTRIBUTES
from mibici.modeling.predict import FORECAST_COLUMN
from mibici.serving.schemas import Instance
from mibici.validation import PanelValidator

# Columnas de estaciones.csv que van al catalogo y al perfil de DemandFeatureBuilder.
CATALOG_COLUMNS = ["station_id", "lat", "lon", "capacity", "zone"]
PROFILE_COLUMNS = ["station_id", "pc1", "pc2", "pc3", "station_cluster"]


class UnknownStationError(LookupError):
    """La peticion trae estaciones que no estan en estaciones.csv."""

    def __init__(self, station_ids: list[int]):
        self.station_ids = station_ids
        super().__init__(f"Estaciones que no existen en {SERVING_STATIONS_FILE}: {station_ids}")


def read_stations(path: Path) -> pd.DataFrame:
    """Lee estaciones.csv con los mismos tipos del panel.

    pandas recupera exactamente los decimales de lat, lon y el PCA que escribio to_csv,
    asi que el pronostico del servicio es igual al que se obtiene con el panel.
    """
    return pd.read_csv(
        path,
        usecols=STATION_ATTRIBUTES,
        dtype={"zone": "str", "station_cluster": "str"},
    )


class ForecastService:
    """Carga el modelo exportado y pronostica a partir de las instancias de la API."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        logger.info(f"Cargando modelo desde {self.model_dir}...")
        info_text = (self.model_dir / SERVING_INFO_FILE).read_text(encoding="utf-8")
        self.info: dict = json.loads(info_text)
        self.model_name: str = self.info["registered_model_name"]
        self.version = int(self.info["version"])
        self.model_uri: str = self.info["model_uri"]
        self.stations = read_stations(self.model_dir / SERVING_STATIONS_FILE)
        self.feature_builder = DemandFeatureBuilder(self.stations[CATALOG_COLUMNS])
        self.model: Pipeline = mlflow.sklearn.load_model(
            str(self.model_dir / SERVING_MODEL_SUBDIR)
        )
        logger.info(
            f"Modelo listo: {self.model_uri} ({self.info.get('algoritmo', 'sin etiqueta')}), "
            f"{len(self.stations)} estaciones"
        )

    def build_panel(self, instances: list[Instance]) -> pd.DataFrame:
        """Arma las columnas que recibe el pipeline, en el orden de las instancias.

        El calendario sale de DemandFeatureBuilder.add_calendar y los atributos de la
        estacion de estaciones.csv con DemandFeatureBuilder.add_station_info, igual que
        al construir el panel de entrenamiento.
        """
        rows = []
        for instance in instances:
            rows.append(instance.model_dump())
        panel = pd.DataFrame(rows)
        panel["hour_ts"] = pd.to_datetime(panel["hour_ts"])

        known = set(self.stations["station_id"])
        unknown = []
        for station_id in panel["station_id"].unique():
            if station_id not in known:
                unknown.append(int(station_id))
        if unknown:
            raise UnknownStationError(sorted(unknown))

        panel = self.feature_builder.add_calendar(panel)
        panel = self.feature_builder.add_station_info(panel, self.stations[PROFILE_COLUMNS])
        return panel

    def predict(self, instances: list[Instance]) -> pd.DataFrame:
        """Devuelve station_id, hour_ts y pronostico por instancia.

        Lanza UnknownStationError si alguna estacion no existe y DataValidationError si el
        panel no pasa PanelValidator (por ejemplo, una estacion-hora repetida).
        """
        panel = self.build_panel(instances)
        validator = PanelValidator()
        validator.validate(panel, require_target=False)
        validator.raise_if_failed()

        out = panel[["station_id", "hour_ts"]].copy()
        out[FORECAST_COLUMN] = clip_counts(self.model.predict(panel[MODEL_FEATURES]))
        logger.info(f"{len(out)} pronosticos con {self.model_uri}")
        return out
