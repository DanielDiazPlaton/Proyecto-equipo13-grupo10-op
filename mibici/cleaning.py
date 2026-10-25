"""Reglas de limpieza de viajes. Cada regla deja registro de cuantas filas afecta."""

from loguru import logger
import numpy as np
import pandas as pd

from mibici.config import CleaningConfig


class TripCleaner:
    """Aplica las reglas de limpieza en orden y genera un reporte por regla.

    Criterio general: se eliminan viajes que no representan demanda real
    (duplicados, viajes cancelados, outliers de duracion, estaciones fuera del
    catalogo). Los errores en datos demograficos no eliminan el viaje, solo se
    marcan como faltantes, porque el viaje si ocurrio.
    """

    def __init__(self, stations: pd.DataFrame, config: CleaningConfig | None = None):
        self.config = config or CleaningConfig()
        self.valid_station_ids = set(stations["station_id"])
        self.report: list[dict] = []

    def _log(self, rule: str, before: int, after: int, note: str = "") -> None:
        removed = before - after
        self.report.append(
            {
                "regla": rule,
                "filas_antes": before,
                "filas_eliminadas": removed,
                "filas_despues": after,
                "nota": note,
            }
        )
        logger.info(f"{rule}: -{removed:,} filas")

    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        cols = [c for c in df.columns if c != "source_month"]
        df = df.drop_duplicates(subset=cols)
        self._log("duplicados_exactos", n, len(df))
        return df

    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df.assign(
            start_time=pd.to_datetime(df["start_time"], errors="coerce"),
            end_time=pd.to_datetime(df["end_time"], errors="coerce"),
        )
        df = df.dropna(subset=["start_time", "end_time"])
        df["duration_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
        df = df[df["duration_min"] >= 0]
        self._log("fechas_invalidas_o_negativas", n, len(df))
        return df

    def drop_short_trips(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df[df["duration_min"] >= self.config.min_duration_min]
        self._log("duracion_menor_1_min", n, len(df), "anclaje fallido o arrepentimiento")
        n = len(df)
        same = df["origin_id"] == df["destination_id"]
        short = df["duration_min"] < self.config.same_station_min_duration_min
        df = df[~(same & short)]
        self._log("misma_estacion_menor_2_min", n, len(df), "bicicleta devuelta de inmediato")
        return df

    def drop_long_trips(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df[df["duration_min"] <= self.config.max_duration_min]
        self._log("duracion_mayor_180_min", n, len(df), "outlier, bicicleta no anclada")
        return df

    def drop_unknown_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        mask = df["origin_id"].isin(self.valid_station_ids) & df["destination_id"].isin(
            self.valid_station_ids
        )
        df = df[mask]
        self._log("estacion_fuera_de_catalogo", n, len(df))
        return df

    def fix_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige genero y ano de nacimiento sin eliminar filas."""
        cfg = self.config
        gender = df["gender"].str.strip().str.upper()
        df["gender"] = gender.where(gender.isin(cfg.valid_genders), "NE").astype("category")
        valid_year = df["birth_year"].between(cfg.min_birth_year, cfg.max_birth_year)
        invalid = int((~valid_year & df["birth_year"].notna()).sum())
        df["birth_year"] = df["birth_year"].where(valid_year)
        df["age"] = cfg.reference_year - df["birth_year"]
        df["age_group"] = pd.cut(
            df["age"],
            bins=[15, 24, 34, 44, 59, 90],
            labels=["16-24", "25-34", "35-44", "45-59", "60+"],
        )
        self.report.append(
            {
                "regla": "anio_nacimiento_invalido_a_nulo",
                "filas_antes": len(df),
                "filas_eliminadas": 0,
                "filas_despues": len(df),
                "nota": f"{invalid} valores fuera de rango convertidos a nulo",
            }
        )
        return df

    def add_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["start_hour"] = df["start_time"].dt.floor("h")
        df["is_round_trip"] = (df["origin_id"] == df["destination_id"]).astype(np.int8)
        return df

    def clean(self, trips: pd.DataFrame) -> pd.DataFrame:
        self.report = []
        df = trips.copy()
        for step in (
            self.drop_duplicates,
            self.parse_dates,
            self.drop_short_trips,
            self.drop_long_trips,
            self.drop_unknown_stations,
            self.fix_demographics,
            self.add_time_columns,
        ):
            df = step(df)
        return df.reset_index(drop=True)

    def report_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.report)


def iqr_upper_fence(values: pd.Series, log_scale: bool = True) -> float:
    """Limite superior de Tukey. En escala log porque la duracion tiene cola larga."""
    x = np.log1p(values) if log_scale else values
    q1, q3 = x.quantile([0.25, 0.75])
    fence = q3 + 1.5 * (q3 - q1)
    return float(np.expm1(fence)) if log_scale else float(fence)
