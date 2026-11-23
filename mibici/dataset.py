"""Carga y descarga de los datos crudos de MiBici (viajes mensuales y catalogo de estaciones)."""

import io
from pathlib import Path
import re
from urllib.parse import urljoin
import zipfile

from bs4 import BeautifulSoup
from loguru import logger
import pandas as pd
import requests
from tqdm.auto import tqdm

from mibici.config import CSV_ENCODINGS, MONTHS, RAW_DATA_DIR, STATIONS_FILE

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

BASE_URL = "https://mibici.net/en/open-data/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def read_csv_any_encoding(path: Path, **kwargs) -> pd.DataFrame:
    """Lee un CSV probando las codificaciones conocidas, en orden."""
    for encoding in CSV_ENCODINGS[:-1]:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            logger.warning(f"{Path(path).name} no esta en {encoding}; se prueba la siguiente")
    return pd.read_csv(path, encoding=CSV_ENCODINGS[-1], **kwargs)


class MiBiciLoader:
    """Lee los CSV publicados en mibici.net y homologa nombres y tipos."""

    def __init__(self, raw_dir: Path = RAW_DATA_DIR):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._catalog: dict[tuple[int, int], str] | None = None

    def trip_file(self, month: str) -> Path:
        return self.raw_dir / f"datos_abiertos_{month}.csv"

    def fetch_catalog(self) -> dict[tuple[int, int], str]:
        """Escanea la página de MiBici y mapea (año, mes) con su URL dinamica."""
        if self._catalog is not None:
            return self._catalog

        logger.info(f"Obteniendo indice de descargas desde {BASE_URL}...")
        res = requests.get(BASE_URL, headers=HEADERS, timeout=30)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, "html.parser")
        catalog = {}
        pattern = re.compile(r"datos_abiertos_(\d{4})_(\d{1,2})\.(csv|zip)", re.IGNORECASE)

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            match = pattern.search(href)
            if match:
                y, m = int(match.group(1)), int(match.group(2))
                catalog[(y, m)] = urljoin(BASE_URL, href)

        self._catalog = catalog
        return self._catalog

    def download_trips(self, year: int, months: list[int] | int) -> list[Path]:
        """
        Descarga los viajes mensuales indicando el año y los meses correspondientes.
        """
        if isinstance(months, int):
            months = [months]

        catalog = self.fetch_catalog()
        downloaded = []

        for m in months:
            target_name = f"datos_abiertos_{year}_{m:02d}.csv"
            target_path = self.raw_dir / target_name

            if target_path.exists():
                logger.info(f"{target_name} ya existe. Omitiendo descarga.")
                downloaded.append(target_path)
                continue

            url = catalog.get((year, m))
            if not url:
                logger.error(f"No se encontró enlace para {year}-{m:02d}.")
                continue

            logger.info(f"Descargando {target_name}...")
            resp = requests.get(url, headers=HEADERS, stream=True, timeout=60)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            buffer = io.BytesIO()

            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc=f"datos_{year}_{m:02d}"
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        buffer.write(chunk)
                        pbar.update(len(chunk))

            buffer.seek(0)

            if url.lower().endswith(".zip") or zipfile.is_zipfile(buffer):
                with zipfile.ZipFile(buffer) as z:
                    csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
                    if csv_files:
                        with z.open(csv_files[0]) as src, open(target_path, "wb") as dst:
                            dst.write(src.read())
                        downloaded.append(target_path)
            else:
                with open(target_path, "wb") as dst:
                    dst.write(buffer.getvalue())
                downloaded.append(target_path)

            logger.success(f"Archivo listo en: {target_path}")

        return downloaded

    def download_stations(self, filename: str = STATIONS_FILE) -> Path:
        """
        Descarga el catalogo / nomenclatura de estaciones desde mibici.net.
        """
        target_path = self.raw_dir / filename
        if target_path.exists():
            logger.info(f"{filename} ya existe. Omitiendo descarga.")
            return target_path

        logger.info("Buscando enlace de nomenclatura de estaciones...")
        res = requests.get(BASE_URL, headers=HEADERS, timeout=30)
        res.raise_for_status()

        soup = BeautifulSoup(res.text, "html.parser")
        station_url = None

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            text = a.get_text().lower()
            if "nomenclatura" in href.lower() or "nomenclatura" in text or "estacion" in text:
                station_url = urljoin(BASE_URL, href)
                break

        if not station_url:
            raise FileNotFoundError("No se encontró el enlace de estaciones en el portal.")

        logger.info(f"Descargando estaciones desde {station_url}...")
        resp = requests.get(station_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        with open(target_path, "wb") as f:
            f.write(resp.content)

        logger.success(f"Catálogo de estaciones guardado en: {target_path}")
        return target_path

    def load_trips(self, months: list[str] = MONTHS) -> pd.DataFrame:
        frames = []
        for month in months:
            path = self.trip_file(month)
            logger.info(f"Leyendo {path.name}")
            df = read_csv_any_encoding(path, dtype={"Genero": "string"})
            df = df.rename(columns=TRIP_COLUMNS)
            df["source_month"] = month
            frames.append(df)
        trips = pd.concat(frames, ignore_index=True)
        logger.info(f"Viajes cargados: {len(trips):,}")
        return trips

    def load_stations(self, filename: str = STATIONS_FILE) -> pd.DataFrame:
        stations = read_csv_any_encoding(self.raw_dir / filename)
        stations = stations.rename(columns=STATION_COLUMNS)
        # El prefijo del nombre indica el municipio: GDL, ZPN o TLQ.
        stations["zone"] = stations["station_name"].str.extract(r"\((GDL|ZPN|TLQ)-")[0]
        stations["is_retired"] = stations["station_name"].str.lower().str.startswith("ex")
        return stations
