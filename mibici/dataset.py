"""Carga de los datos crudos de MiBici (viajes mensuales y catalogo de estaciones)."""

from pathlib import Path

from loguru import logger
import pandas as pd

from mibici.config import MONTHS, RAW_DATA_DIR, STATIONS_FILE

TRIP_COLUMNS = {
    "Viaje_Id": "trip_id",
    "Usuario_Id": "user_id",
    "Genero": "gender",
    "Año_de_nacimiento": "birth_year",
    "Inicio_del_viaje": "start_time",
    "Fin_del_viaje": "end_time",
    "Origen_Id": "origin_id",
    "Destino_Id": "destination_id",
}

STATION_COLUMNS = {
    "id": "station_id",
    "name": "station_name",
    "latitude": "lat",
    "longitude": "lon",
    "dpcapacity": "capacity",
}


class MiBiciLoader:
    """Lee los CSV publicados en mibici.net y homologa nombres y tipos."""

    def __init__(self, raw_dir: Path = RAW_DATA_DIR):
        self.raw_dir = Path(raw_dir)

    def trip_file(self, month: str) -> Path:
        return self.raw_dir / f"datos_abiertos_{month}.csv"

    def load_trips(self, months: list[str] = MONTHS) -> pd.DataFrame:
        frames = []
        for month in months:
            path = self.trip_file(month)
            logger.info(f"Leyendo {path.name}")
            df = pd.read_csv(path, encoding="utf-8", dtype={"Genero": "string"})
            df = df.rename(columns=TRIP_COLUMNS)
            df["source_month"] = month
            frames.append(df)
        trips = pd.concat(frames, ignore_index=True)
        logger.info(f"Viajes cargados: {len(trips):,}")
        return trips

    def load_stations(self, filename: str = STATIONS_FILE) -> pd.DataFrame:
        stations = pd.read_csv(self.raw_dir / filename, encoding="utf-8")
        stations = stations.rename(columns=STATION_COLUMNS)
        # El prefijo del nombre indica el municipio: GDL, ZPN o TLQ.
        stations["zone"] = stations["station_name"].str.extract(r"\((GDL|ZPN|TLQ)-")[0]
        stations["is_retired"] = stations["station_name"].str.lower().str.startswith("ex")
        return stations
