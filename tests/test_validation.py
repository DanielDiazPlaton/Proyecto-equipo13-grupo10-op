"""Pruebas de la validacion de datos: panel estacion-hora y CSV crudos."""

import pandas as pd
import pytest

from mibici import validation
from mibici.config import STATIONS_FILE, TrainingConfig
from mibici.features import TARGET
from mibici.modeling.train import DemandExperiment
from mibici.validation import REPORT_COLUMNS, DataValidationError, PanelValidator, RawDataValidator
from tests.conftest import TEST_START


def test_valid_panel_passes_and_reports_every_rule(panel, tmp_path):
    validator = PanelValidator()
    report = validator.validate(panel)
    assert validator.status == "ok"
    assert validator.stats["filas"] == len(panel)
    assert list(report.columns) == REPORT_COLUMNS
    assert (report["resultado"] == "ok").all()
    # Al pronosticar el panel puede no traer la variable objetivo.
    validator.validate(panel.drop(columns=[TARGET]), require_target=False)
    assert validator.ok
    saved = pd.read_csv(validator.save_report(tmp_path / "reporte.csv"))
    assert len(saved) == len(validator.report)


def _negative_target(df):
    df.loc[df.index[0], TARGET] = -1
    return df


def _duplicated_row(df):
    return pd.concat([df, df.head(1)], ignore_index=True)


def _cluster_as_int(df):
    # El modelo se entreno con station_cluster como texto; como numero el encoder falla con
    # un TypeError poco claro.
    df["station_cluster"] = df["station_cluster"].astype(int)
    return df


@pytest.mark.parametrize(
    "corrupt, rule",
    [
        (_negative_target, "objetivo_no_negativo"),
        (_duplicated_row, "llave_sin_duplicados"),
        (_cluster_as_int, "tipos_de_dato"),
    ],
)
def test_schema_errors_stop_the_process(panel, corrupt, rule):
    validator = PanelValidator()
    validator.validate(corrupt(panel.copy()))
    assert [row["regla"] for row in validator.errors] == [rule]
    with pytest.raises(DataValidationError, match=rule):
        validator.raise_if_failed()


def test_plausibility_issues_are_only_warnings(panel):
    df = panel.copy()
    df.loc[df["station_id"] == 1, "lat"] = 0.0  # fuera de Guadalajara
    validator = PanelValidator()
    validator.validate(df, expected_fingerprint="0" * 64)
    assert validator.status == "ok_con_advertencias"
    assert {row["regla"] for row in validator.warnings} == {"coordenadas_en_area", "huella_panel"}


def test_experiment_rejects_invalid_panel_before_mlflow(panel, tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    with pytest.raises(DataValidationError, match="objetivo_no_negativo"):
        DemandExperiment(
            _negative_target(panel.copy()), TrainingConfig(test_start=TEST_START), tracking_uri=uri
        )
    assert not (tmp_path / "mlflow.db").exists()  # no se creo ningun run


def write_raw_files(raw_dir, stations_file):
    """Un mes de viajes con una fila y un catalogo de dos estaciones en Latin-1."""
    raw_dir.mkdir()
    trips = pd.DataFrame(
        [[1, 10, "M", 1990, "2026-07-01 08:00:00", "2026-07-01 08:10:00", 1, 2]],
        columns=list(validation.TRIP_COLUMNS),
    )
    trips.to_csv(raw_dir / "datos_abiertos_2026_07.csv", index=False)
    stations = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["(GDL-001) Av. Juárez", "(ZPN-002) Plaza Patria"],
            "latitude": [20.67, 20.71],
            "longitude": [-103.35, -103.38],
            "dpcapacity": [12, 15],
        }
    )
    stations.to_csv(raw_dir / stations_file, index=False, encoding="latin-1")


def test_raw_validator_checks_files(tmp_path):
    validator = RawDataValidator(tmp_path / "vacio", ["2026_07"], "estaciones.csv")
    validator.validate()
    assert [row["regla"] for row in validator.errors] == ["archivos_presentes"]

    raw_dir = tmp_path / "raw"
    write_raw_files(raw_dir, "estaciones.csv")
    validator = RawDataValidator(raw_dir, ["2026_07"], "estaciones.csv")
    validator.validate()
    assert validator.ok, validator.errors
    assert validator.stats["filas_v0"] == 1

    # Un catalogo sin la columna name se reporta en la regla, sin que el proceso se caiga.
    stations = pd.read_csv(raw_dir / "estaciones.csv", encoding="latin-1")
    stations.drop(columns=["name"]).to_csv(raw_dir / "estaciones.csv", index=False)
    validator.validate()
    assert [row["regla"] for row in validator.errors] == ["catalogo_columnas"]


def test_raw_validator_detects_a_different_catalog(tmp_path, monkeypatch):
    # La descarga automatica guarda cualquier catalogo con el nombre del proyecto.
    raw_dir = tmp_path / "raw"
    write_raw_files(raw_dir, STATIONS_FILE)
    validator = RawDataValidator(raw_dir, ["2026_07"])
    validator.validate()
    assert [row["regla"] for row in validator.errors] == ["catalogo_estaciones"]

    # Mismo numero de estaciones pero otro contenido: lo detecta la huella.
    monkeypatch.setattr(validation, "EXPECTED_STATIONS", 2)
    validator.validate()
    assert [row["regla"] for row in validator.errors] == ["catalogo_huella"]
