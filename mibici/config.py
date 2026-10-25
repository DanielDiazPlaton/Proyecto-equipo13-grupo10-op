"""Configuracion central del proyecto: rutas, parametros de limpieza y de MLflow."""

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

# MLflow: backend local en SQLite para que cualquier integrante pueda
# reproducir los runs sin levantar un servidor. Se puede cambiar con la
# variable de entorno MLFLOW_TRACKING_URI.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "mibici-fase1-datos"

# Meses usados en la Fase 1
MONTHS = ["2026_07", "2026_08", "2026_09"]
STATIONS_FILE = "nomenclatura_2026_09.csv"

# Dias festivos dentro del periodo analizado (Mexico)
HOLIDAYS = ["2026-09-16"]


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
    random_state: int = 42

    def to_params(self) -> dict:
        params = asdict(self)
        params["lags"] = ",".join(str(x) for x in self.lags)
        return params
