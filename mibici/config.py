"""Configuracion central del proyecto: rutas, parametros de datos, de modelado y de MLflow.

Todo valor que cambia entre corridas (rutas, umbrales, semillas, fechas de corte,
hiperparametros) vive aqui. Los modulos lo reciben como parametro y lo registran en MLflow.
"""

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Rutas (estructura CookieCutter Data Science)
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Archivos que conectan las etapas del pipeline
CLEAN_TRIPS_FILE = INTERIM_DATA_DIR / "viajes_limpios_v1.parquet"
PANEL_FILE = PROCESSED_DATA_DIR / "panel_estacion_hora_v2.parquet"
PREDICTIONS_FILE = PROCESSED_DATA_DIR / "pronostico_salidas.parquet"
# v2.1: mismo panel que la v2 de la Fase 1, con los atributos de estacion del
# catalogo de octubre de 2026 (ver references/bitacora_cambios_datos.md).
PANEL_VERSION = "v2.1"

# MLflow: backend local en SQLite para que cualquier integrante pueda
# reproducir los runs sin levantar un servidor. Se puede cambiar con la
# variable de entorno MLFLOW_TRACKING_URI.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "mibici-fase1-datos"
MLFLOW_MODEL_EXPERIMENT = "mibici-fase2-modelos"

# Model Registry: un solo modelo registrado; cada candidato es una version.
REGISTERED_MODEL_NAME = "mibici-demanda-salidas"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
# MLflow guarda los pipelines en formato skops, que solo carga tipos declarados
# como confiables. Estos son los que usan los modelos candidatos.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

# Meses usados en el proyecto
MONTHS = ["2026_07", "2026_08", "2026_09"]
STATIONS_FILE = "nomenclatura_2026_10.csv"
# MiBici publica unos archivos en UTF-8 y otros en Latin-1; se prueban en este orden.
CSV_ENCODINGS = ("utf-8", "latin-1")

# Dias festivos dentro del periodo analizado (Mexico)
HOLIDAYS = ["2026-09-16"]

# Semilla unica y fecha de corte de la particion temporal. Entrenamiento: antes
# de TEST_START. Prueba: desde TEST_START hasta el final del periodo.
RANDOM_STATE = 42
TEST_START = "2026-09-17"


@dataclass
class CleaningConfig:
    """Umbrales de limpieza. Se registran como parametros en MLflow."""

    min_duration_min: float = 1.0
    same_station_min_duration_min: float = 2.0
    max_duration_min: float = 180.0
    min_birth_year: int = 1940
    max_birth_year: int = 2010
    reference_year: int = 2026
    valid_genders: tuple = ("M", "F")

    def to_params(self) -> dict:
        params = asdict(self)
        params["valid_genders"] = ",".join(self.valid_genders)
        return params


@dataclass
class FeatureConfig:
    """Parametros de construccion del panel estacion-hora."""

    lags: list = field(default_factory=lambda: [1, 24, 168])
    rolling_window: int = 24
    n_pca_components: int = 3
    k_clusters: int = 3
    random_state: int = RANDOM_STATE

    def to_params(self) -> dict:
        params = asdict(self)
        params["lags"] = ",".join(str(x) for x in self.lags)
        return params


# Espacios de busqueda de hiperparametros por modelo. Las llaves usan la
# convencion de scikit-learn "<paso>__<parametro>" del Pipeline. La busqueda
# aleatoria toma n_iter combinaciones de cada espacio.
SEARCH_SPACES: dict[str, dict[str, list]] = {
    "ridge": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    },
    "poisson_glm": {
        "model__alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    "gradient_boosting": {
        "model__loss": ["poisson", "squared_error"],
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__max_iter": [100, 200, 300],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [20, 50, 100, 200],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    },
}


@dataclass
class TrainingConfig:
    """Parametros del entrenamiento, el ajuste y la evaluacion de modelos."""

    target: str = "departures"
    test_start: str = TEST_START
    models: tuple = ("ridge", "poisson_glm", "gradient_boosting")
    # Validacion cruzada temporal de ventana expandible sobre el periodo de entrenamiento
    cv_splits: int = 3
    # Combinaciones de hiperparametros que se prueban por modelo
    n_iter: int = 8
    # Metrica con la que se elige la mejor combinacion y el modelo campeon
    selection_metric: str = "rmse"
    n_jobs: int = 1
    random_state: int = RANDOM_STATE
    # Filas de prueba usadas para la importancia por permutacion
    importance_sample: int = 50_000

    def to_params(self) -> dict:
        params = asdict(self)
        params["models"] = ",".join(self.models)
        return params
