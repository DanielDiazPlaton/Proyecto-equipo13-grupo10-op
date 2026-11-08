"""Validación de los datos antes de usarlos: CSV crudos y panel estación-hora.

Hay dos puntos de control:

1. Datos crudos, antes de ejecutar ``make data``. Se verifica que los CSV que se descargan
   de mibici.net (con MiBiciLoader.download_trips y MiBiciLoader.download_stations, o que se
   descargan a mano) estén completos, que tengan todas las columnas que necesita el
   cargador, que la huella coincida con la versión v0 registrada en MLflow y que el catálogo
   de estaciones sea el del proyecto.

2. Panel estación-hora, antes de entrenar o predecir. Se verifica que el panel tenga el
   esquema que esperan los pipelines de scikit-learn: columnas correctas, tipos correctos,
   clave estación-hora sin duplicados, variable objetivo válida y variables construidas
   dentro del rango permitido. Los comandos ``make data``, ``make train`` y ``make predict``
   hacen esta verificación de forma automática y se detienen si encuentran un error.

Los resultados se dividen en errores y advertencias. Los errores detienen el proceso porque
el modelo entrenaría o predeciría con datos equivocados. Las advertencias se registran pero
no impiden que el proceso continúe (por ejemplo, una huella distinta por diferencias en los
últimos decimales del PCA entre equipos).

Igual que TripCleaner en la limpieza, cada regla deja un renglón en un reporte (regla,
resultado, casos y nota) que se guarda en reports/tables.

Uso:
    python -m mibici.validation raw      # CSV en data/raw (make check-raw)
    python -m mibici.validation panel    # panel en data/processed (make validate)
"""

import hashlib
from pathlib import Path

from loguru import logger
import pandas as pd
from pandas.api import types as ptypes
import typer

from mibici.config import (
    EXPECTED_FINGERPRINTS,
    EXPECTED_RAW_ROWS,
    EXPECTED_STATIONS,
    EXPECTED_STATIONS_FILE_SHA256,
    MONTHS,
    PANEL_FILE,
    PANEL_VERSION,
    RAW_DATA_DIR,
    STATIONS_FILE,
    TABLES_DIR,
    PanelValidationConfig,
)
from mibici.dataset import STATION_COLUMNS, TRIP_COLUMNS, MiBiciLoader, read_csv_any_encoding
from mibici.features import (
    CATEGORICAL_FEATURES,
    COUNT_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)
from mibici.versioning import dataframe_fingerprint

app = typer.Typer()

DOWNLOAD_PAGE = "https://www.mibici.net/es/datos-abiertos/"
KEY_COLUMNS = ["station_id", "hour_ts"]
BINARY_FEATURES = ["is_weekend", "is_holiday"]
CYCLIC_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
# Errores de redondeo permitidos al revisar que el seno y el coseno estén en [-1, 1]
CYCLIC_TOLERANCE = 1e-9

# Resultado de cada regla en el reporte
OK, ERROR, WARNING = "ok", "error", "advertencia"
REPORT_COLUMNS = ["regla", "resultado", "casos", "nota"]

# Reportes que dejan make check-raw, make data y make validate
RAW_REPORT_FILE = TABLES_DIR / "reporte_validacion_crudos.csv"
PANEL_REPORT_FILE = TABLES_DIR / "reporte_validacion_panel.csv"


class DataValidationError(ValueError):
    """Los datos no pasaron la validación. El reporte completo queda en .report."""

    def __init__(self, name: str, report: pd.DataFrame):
        self.report = report
        errors = report[report["resultado"] == ERROR]
        detail = "\n- ".join(f"{row.regla}: {row.nota}" for row in errors.itertuples())
        super().__init__(f"{name}: {len(errors)} error(es) de validacion:\n- {detail}")


class DataValidator:
    """Aplica reglas en orden y genera un reporte por regla.

    Es la base de PanelValidator y RawDataValidator. Cada regla deja un renglón con su
    resultado (ok, error o advertencia), cuántos casos encontró y una nota.
    """

    name = "datos"

    def __init__(self):
        self.report: list[dict] = []
        self.stats: dict = {}

    def _log(self, rule: str, cases: int, note: str = "", result: str = ERROR) -> None:
        """Deja el renglón de la regla: ok si no encontró casos; si los encontró, su resultado."""
        if not cases:
            self.report.append({"regla": rule, "resultado": OK, "casos": 0, "nota": ""})
            logger.info(f"{rule}: ok")
            return
        self.report.append({"regla": rule, "resultado": result, "casos": cases, "nota": note})
        if result == WARNING:
            logger.warning(f"{rule}: {note}")
        else:
            logger.error(f"{rule}: {note}")

    def _log_result(self) -> None:
        n_errors, n_warnings = len(self.errors), len(self.warnings)
        if n_errors:
            logger.error(
                f"Validacion de {self.name}: {n_errors} errores, {n_warnings} advertencias"
            )
        else:
            logger.success(f"Validacion de {self.name}: {self.status}")

    @property
    def errors(self) -> list[dict]:
        return [row for row in self.report if row["resultado"] == ERROR]

    @property
    def warnings(self) -> list[dict]:
        return [row for row in self.report if row["resultado"] == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def status(self) -> str:
        """Etiqueta corta para MLflow: ok, ok_con_advertencias o con_errores."""
        if self.errors:
            return "con_errores"
        return "ok_con_advertencias" if self.warnings else "ok"

    def report_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.report, columns=REPORT_COLUMNS)

    def save_report(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.report_frame().to_csv(path, index=False)
        logger.info(f"Reporte de validacion guardado en: {path}")
        return path

    def raise_if_failed(self) -> None:
        if self.errors:
            raise DataValidationError(self.name, self.report_frame())


def panel_schema(require_target: bool = True) -> dict[str, str]:
    """Columnas que necesita el modelado y el tipo de dato que debe tener cada una.

    Se arma con las mismas listas de variables que usa el preprocesamiento, de modo
    que si cambian las variables del modelo la validación cambia con ellas.
    """
    schema = {"station_id": "entero", "hour_ts": "fecha", "hour": "entero"}
    if require_target:
        schema[TARGET] = "entero"
    for column in NUMERIC_FEATURES:
        schema[column] = "entero" if column in BINARY_FEATURES else "numerico"
    for column in CATEGORICAL_FEATURES:
        schema[column] = "texto"
    return schema


def has_type(values: pd.Series, kind: str) -> bool:
    """Revisa el tipo de una columna: entero, numerico, fecha o texto."""
    if kind == "entero":
        return ptypes.is_integer_dtype(values)
    if kind == "numerico":
        return ptypes.is_numeric_dtype(values) and not ptypes.is_bool_dtype(values)
    if kind == "fecha":
        return ptypes.is_datetime64_any_dtype(values)
    return ptypes.is_string_dtype(values)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Huella SHA-256 del archivo tal como se descargó (bytes, no contenido leído)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PanelValidator(DataValidator):
    """Valida el panel estación-hora contra el esquema que esperan los modelos."""

    name = "panel"

    def __init__(self, config: PanelValidationConfig | None = None):
        super().__init__()
        self.config = config or PanelValidationConfig()

    def check_columns(self, panel: pd.DataFrame, schema: dict[str, str]) -> None:
        missing = [c for c in schema if c not in panel.columns]
        self._log("columnas_requeridas", len(missing), f"Faltan columnas: {missing}")
        self._log("panel_con_filas", int(panel.empty), "El panel no tiene filas")

    def check_types(self, panel: pd.DataFrame, schema: dict[str, str]) -> None:
        wrong = [
            f"{column} debe ser de tipo {kind} y es {panel[column].dtype}"
            for column, kind in schema.items()
            if not has_type(panel[column], kind)
        ]
        self._log("tipos_de_dato", len(wrong), "; ".join(wrong))

    def check_keys(self, panel: pd.DataFrame) -> None:
        """Cada fila es una estación en una hora exacta, sin repetirse."""
        nulls = int(panel[KEY_COLUMNS].isna().any(axis=1).sum())
        self._log("llave_sin_nulos", nulls, f"{nulls:,} filas sin station_id u hour_ts")
        duplicated = int(panel.duplicated(subset=KEY_COLUMNS).sum())
        self._log(
            "llave_sin_duplicados",
            duplicated,
            f"Hay {duplicated:,} filas repetidas de estacion-hora",
        )
        ts = panel["hour_ts"]
        off_hour = int((ts.notna() & (ts != ts.dt.floor("h"))).sum())
        self._log(
            "horas_exactas",
            off_hour,
            f"hour_ts tiene {off_hour:,} valores que no son una hora exacta",
        )
        mismatch = int((ts.notna() & (panel["hour"] != ts.dt.hour)).sum())
        self._log(
            "hora_consistente", mismatch, f"hour no coincide con hour_ts en {mismatch:,} filas"
        )

    def check_target(self, panel: pd.DataFrame) -> None:
        """La variable objetivo es un conteo: sin nulos y sin valores negativos."""
        nulls = int(panel[TARGET].isna().sum())
        self._log("objetivo_sin_nulos", nulls, f"{TARGET} tiene {nulls:,} nulos")
        negative = int((panel[TARGET] < 0).sum())
        self._log(
            "objetivo_no_negativo", negative, f"{TARGET} tiene {negative:,} valores negativos"
        )

    def check_ranges(self, panel: pd.DataFrame) -> None:
        """Variables construidas dentro de su rango."""
        problems = []
        total = 0
        for column in BINARY_FEATURES:
            invalid = int((panel[column].notna() & ~panel[column].isin([0, 1])).sum())
            if invalid:
                problems.append(f"{column} tiene {invalid:,} valores distintos de 0 y 1")
                total += invalid
        self._log("banderas_0_o_1", total, ", ".join(problems))

        problems = []
        total = 0
        for column in CYCLIC_FEATURES:
            outside = int((panel[column].abs() > 1 + CYCLIC_TOLERANCE).sum())
            if outside:
                problems.append(f"{column} tiene {outside:,} valores fuera de [-1, 1]")
                total += outside
        self._log("ciclicas_en_rango", total, ", ".join(problems))

        problems = []
        total = 0
        for column in COUNT_FEATURES:
            negative = int((panel[column] < 0).sum())
            if negative:
                problems.append(f"{column} tiene {negative:,} valores negativos")
                total += negative
        self._log("conteos_no_negativos", total, ", ".join(problems))

    def check_nulls(self, panel: pd.DataFrame) -> None:
        """Nulos por variable: arriba del límite es error; abajo, el pipeline los imputa."""
        limit = self.config.max_null_pct
        over_columns = []
        over_notes = []
        under_columns = []
        under_notes = []
        for column in MODEL_FEATURES:
            pct = float(panel[column].isna().mean() * 100)
            if pct > limit:
                over_columns.append(column)
                over_notes.append(f"{column} tiene {pct:.1f} % de nulos (limite {limit} %)")
            elif pct > 0:
                under_columns.append(column)
                under_notes.append(
                    f"{column} tiene {pct:.2f} % de nulos; se imputan con la mediana"
                )

        over_rows = 0
        if over_columns:
            over_rows = int(panel[over_columns].isna().any(axis=1).sum())
        self._log("nulos_bajo_el_limite", over_rows, ", ".join(over_notes))

        under_rows = 0
        if under_columns:
            under_rows = int(panel[under_columns].isna().any(axis=1).sum())
        self._log("nulos_imputables", under_rows, ", ".join(under_notes), WARNING)

    def check_stations(self, panel: pd.DataFrame) -> None:
        """Atributos de estación poco creíbles. Son advertencias, no detienen el proceso."""
        cfg = self.config
        stations = panel.drop_duplicates("station_id")
        lat_ok = stations["lat"].between(*cfg.lat_range) | stations["lat"].isna()
        lon_ok = stations["lon"].between(*cfg.lon_range) | stations["lon"].isna()
        outside = stations.loc[~(lat_ok & lon_ok), "station_id"].tolist()
        self._log(
            "coordenadas_en_area",
            len(outside),
            f"{len(outside)} estaciones con coordenadas fuera del area metropolitana: {outside[:10]}",
            WARNING,
        )
        no_capacity = stations.loc[stations["capacity"] <= 0, "station_id"].tolist()
        self._log(
            "capacidad_positiva",
            len(no_capacity),
            f"{len(no_capacity)} estaciones con capacidad no positiva: {no_capacity[:10]}",
            WARNING,
        )
        zones = sorted(set(panel["zone"].dropna().unique()) - set(cfg.valid_zones))
        self._log(
            "municipio_reconocido",
            len(zones),
            f"Municipios no reconocidos en zone: {zones}",
            WARNING,
        )

    def check_fingerprint(self, panel: pd.DataFrame, expected: str | None) -> None:
        """Compara la huella del panel con la registrada. Solo advierte: el PCA cambia decimales."""
        fingerprint = dataframe_fingerprint(panel)
        self.stats["sha256"] = fingerprint
        if expected is None:
            return
        self._log(
            "huella_panel",
            int(fingerprint != expected),
            f"La huella del panel ({fingerprint[:12]}) no coincide con la registrada "
            f"({expected[:12]}). Revisa la bitacora de cambios a los datos.",
            WARNING,
        )

    def validate(
        self,
        panel: pd.DataFrame,
        require_target: bool = True,
        expected_fingerprint: str | None = None,
    ) -> pd.DataFrame:
        """Aplica las reglas en orden y devuelve el reporte por regla (no lanza excepciones).

        require_target=False se usa al pronosticar, cuando el panel puede no traer el
        valor real. expected_fingerprint compara la huella del panel con la registrada.
        Para detener el proceso ante un error se llama después a raise_if_failed().
        """
        self.report, self.stats = [], {}
        logger.info(f"Validando panel ({len(panel):,} filas)...")
        schema = panel_schema(require_target)
        self.check_columns(panel, schema)
        if self.ok:
            self.check_types(panel, schema)
        # Las demás reglas suponen que las columnas existen y tienen el tipo correcto.
        if self.ok:
            self.check_keys(panel)
            if require_target:
                self.check_target(panel)
            self.check_ranges(panel)
            self.check_nulls(panel)
            self.check_stations(panel)
            self.check_fingerprint(panel, expected_fingerprint)
            self.stats.update(
                {
                    "filas": len(panel),
                    "columnas": panel.shape[1],
                    "estaciones": int(panel["station_id"].nunique()),
                    "horas": int(panel["hour_ts"].nunique()),
                    "desde": str(panel["hour_ts"].min()),
                    "hasta": str(panel["hour_ts"].max()),
                }
            )
            logger.info(
                f"Panel: {self.stats['filas']:,} filas, {self.stats['estaciones']} "
                f"estaciones, del {self.stats['desde']} al {self.stats['hasta']}"
            )
        self._log_result()
        return self.report_frame()


class RawDataValidator(DataValidator):
    """Revisa los CSV de data/raw antes de correr el pipeline de datos."""

    name = "datos_crudos"

    def __init__(
        self,
        raw_dir: Path = RAW_DATA_DIR,
        months: list[str] = MONTHS,
        stations_file: str = STATIONS_FILE,
    ):
        super().__init__()
        self.loader = MiBiciLoader(raw_dir)
        self.months = list(months)
        self.stations_file = stations_file

    @property
    def is_project_period(self) -> bool:
        """Las filas y la huella esperadas solo aplican a los meses del proyecto."""
        return self.months == list(MONTHS)

    @property
    def is_project_catalog(self) -> bool:
        """El número de estaciones y la huella esperados solo aplican al catálogo del proyecto."""
        return self.stations_file == STATIONS_FILE

    def expected_files(self) -> dict[str, Path]:
        files = {m: self.loader.trip_file(m) for m in self.months}
        files["estaciones"] = self.loader.raw_dir / self.stations_file
        return files

    def check_files(self) -> None:
        """Que estén todos los archivos. Anota el tamaño y la huella de cada uno."""
        files = self.expected_files()
        missing = [p.name for p in files.values() if not p.exists()]
        self._log(
            "archivos_presentes",
            len(missing),
            f"Faltan en {self.loader.raw_dir}: {missing}. Descargalos con "
            f"MiBiciLoader.download_trips() y download_stations(), o desde {DOWNLOAD_PAGE}",
        )
        if missing:
            return
        self.stats["archivos"] = {}
        for path in files.values():
            info = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            self.stats["archivos"][path.name] = info
            logger.info(f"{path.name}: {info['bytes']:,} bytes, sha256 {info['sha256']}")

    def check_trip_columns(self) -> None:
        problems = []
        for month in self.months:
            path = self.loader.trip_file(month)
            header = read_csv_any_encoding(path, nrows=0).columns
            absent = [c for c in TRIP_COLUMNS if c not in header]
            if absent:
                problems.append(f"{path.name} no tiene las columnas {absent}")
        self._log("columnas_viajes", len(problems), "; ".join(problems))

    def check_catalog(self) -> None:
        # Se revisa el encabezado antes de cargar: load_stations usa la columna name para
        # sacar el municipio y fallaria con KeyError si no viene.
        path = self.loader.raw_dir / self.stations_file
        header = read_csv_any_encoding(path, nrows=0).columns
        absent = []
        for column in STATION_COLUMNS:
            if column not in header:
                absent.append(column)
        self._log(
            "catalogo_columnas",
            len(absent),
            f"{self.stations_file} no tiene las columnas {absent}",
        )
        if absent:
            return
        stations = self.loader.load_stations(self.stations_file)
        duplicated = int(stations["station_id"].duplicated().sum())
        self._log(
            "catalogo_ids_unicos",
            duplicated,
            f"{self.stations_file} tiene {duplicated} ids de estacion repetidos",
        )
        no_zone = int(stations["zone"].isna().sum())
        self._log(
            "catalogo_municipio",
            no_zone,
            f"{no_zone} estaciones sin municipio en el nombre; en el panel quedan como OTRA",
            WARNING,
        )
        self.stats["estaciones_catalogo"] = len(stations)
        if self.is_project_catalog:
            self.check_project_catalog(stations)

    def check_project_catalog(self, stations: pd.DataFrame) -> None:
        """Comprueba que el catálogo sea el mismo con el que se construyó el panel v2.1.

        MiBici solo publica el catálogo más reciente y download_stations lo guarda con el
        nombre STATIONS_FILE. Cuando MiBici publique uno nuevo, el archivo tendrá el nombre
        del catálogo del proyecto con otro contenido.

        Primero se revisa el número de estaciones, lo cual muestra un mensaje fácil de leer,
        si coincide, se revisa la huella del archivo, que detecta cualquier otro cambio
        (capacidades, coordenadas o nombres).
        """
        advice = (
            "Es probable que MiBici haya publicado un catalogo nuevo. Consigue la copia del "
            "catalogo del proyecto o registra una version nueva de los datos"
        )
        n_stations = len(stations)
        self._log(
            "catalogo_estaciones",
            int(n_stations != EXPECTED_STATIONS),
            f"{self.stations_file} tiene {n_stations} estaciones y el catalogo del proyecto "
            f"tiene {EXPECTED_STATIONS}. {advice}",
        )
        if n_stations != EXPECTED_STATIONS:
            return
        digest = file_sha256(self.loader.raw_dir / self.stations_file)
        self._log(
            "catalogo_huella",
            int(digest != EXPECTED_STATIONS_FILE_SHA256),
            f"La huella de {self.stations_file} ({digest[:12]}) no coincide con la del catalogo "
            f"del proyecto ({EXPECTED_STATIONS_FILE_SHA256[:12]}), aunque tiene "
            f"{EXPECTED_STATIONS} estaciones: cambiaron capacidades, coordenadas o nombres. "
            f"{advice}",
        )

    def check_trips(self) -> None:
        """Filas y huella de la versión v0 (los tres meses concatenados)."""
        trips = self.loader.load_trips(self.months)
        fingerprint = dataframe_fingerprint(trips)
        self.stats.update(
            {
                "filas_v0": len(trips),
                "sha256_v0": fingerprint,
                "filas_por_mes": trips["source_month"].value_counts().sort_index().to_dict(),
            }
        )
        if not self.is_project_period:
            return
        rows_differ = len(trips) != EXPECTED_RAW_ROWS
        self._log(
            "viajes_filas",
            int(rows_differ),
            f"Los CSV tienen {len(trips):,} viajes y la version v0 tiene "
            f"{EXPECTED_RAW_ROWS:,}. Es posible que MiBici haya republicado algun mes.",
        )
        if rows_differ:
            return
        expected = EXPECTED_FINGERPRINTS["v0"]
        self._log(
            "viajes_huella_v0",
            int(fingerprint != expected),
            f"La huella de v0 ({fingerprint[:12]}) no coincide con la registrada "
            f"({expected[:12]}): el contenido de los CSV cambio.",
        )

    def validate(self, load_trips: bool = True) -> pd.DataFrame:
        """Aplica las reglas en orden y devuelve el reporte por regla.

        load_trips=False revisa solo los archivos, sus encabezados y el catálogo, sin
        leer los viajes.
        """
        self.report, self.stats = [], {}
        logger.info(f"Revisando datos crudos en {self.loader.raw_dir}...")
        self.check_files()
        if self.ok:
            self.check_trip_columns()
            self.check_catalog()
        if self.ok and load_trips:
            self.check_trips()
        self._log_result()
        return self.report_frame()


@app.command()
def raw(raw_dir: Path = RAW_DATA_DIR):
    """Revisa los CSV descargados antes de correr make data."""
    validator = RawDataValidator(raw_dir)
    validator.validate()
    validator.save_report(RAW_REPORT_FILE)
    if not validator.ok:
        raise typer.Exit(code=1)


@app.command()
def panel(panel_path: Path = PANEL_FILE):
    """Revisa el panel procesado antes de entrenar."""
    if not Path(panel_path).exists():
        raise typer.BadParameter(f"No existe {panel_path}. Corre primero: make data")
    validator = PanelValidator()
    validator.validate(
        pd.read_parquet(panel_path), expected_fingerprint=EXPECTED_FINGERPRINTS[PANEL_VERSION]
    )
    validator.save_report(PANEL_REPORT_FILE)
    if not validator.ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
