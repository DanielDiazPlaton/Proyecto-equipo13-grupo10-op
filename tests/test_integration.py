"""Pruebas de integracion: el flujo completo, de los CSV crudos a las metricas del pronostico.

    carga de datos -> limpieza -> variables -> validacion -> entrenamiento y registro en
    MLflow -> prediccion con el modelo del Model Registry -> metricas

Los CSV son sinteticos, con el formato en que MiBici publica sus datos, y se escriben en un
directorio temporal junto con su propio mlflow.db. Por eso las pruebas no necesitan los datos
reales ni tocan los experimentos del proyecto.

Solo estas pruebas:    pytest -m integration
Todas menos estas:     pytest -m "not integration"
"""

from types import SimpleNamespace

from mlflow import MlflowClient
import numpy as np
import pandas as pd
import pytest

from mibici.cleaning import TripCleaner
from mibici.config import CHAMPION_ALIAS, FeatureConfig, TrainingConfig
from mibici.dataset import TRIP_COLUMNS, MiBiciLoader
from mibici.features import TARGET, DemandFeatureBuilder, StationProfiler
from mibici.modeling.evaluation import NAIVE_BASELINES, regression_metrics, temporal_split
from mibici.modeling.predict import FORECAST_COLUMN, DemandForecaster
from mibici.modeling.train import REFERENCE_BASELINE, DemandExperiment
from mibici.validation import PanelValidator
from mibici.versioning import dataframe_fingerprint

pytestmark = pytest.mark.integration

N_STATIONS = 12
MONTHS = ["2026_08", "2026_09"]
STATIONS_FILE = "nomenclatura.csv"
# Cuatro semanas: la primera solo alimenta el rezago semanal, dos entrenan y una prueba.
PERIOD_START = "2026-08-25"
PERIOD_END = "2026-09-21 23:00"
TEST_START = "2026-09-15"
TEST_HOURS = 7 * 24
MODELS = ("ridge", "gradient_boosting")
SEARCH_SPACES = {
    "ridge": {"model__alpha": [0.1, 10.0]},
    "gradient_boosting": {"model__loss": ["poisson"], "model__max_iter": [20, 40]},
}
REGISTERED_MODEL = "modelo-integracion"


def make_raw_trips(seed: int = 0) -> pd.DataFrame:
    """Viajes validos con los nombres de columna originales de MiBici.

    Las estaciones impares tienen el pico en la manana y las pares en la tarde, para que el
    agrupamiento de perfiles horarios tenga algo que separar.
    """
    rng = np.random.default_rng(seed)
    hours = pd.date_range(PERIOD_START, PERIOD_END, freq="h")
    hour, dow = hours.hour.to_numpy(), hours.day_of_week.to_numpy()
    morning = np.exp(-((hour - 8) ** 2) / 8)
    evening = np.exp(-((hour - 18) ** 2) / 8)
    frames = []
    for station in range(1, N_STATIONS + 1):
        peaks = 1.3 * morning + 0.5 * evening if station % 2 else 0.5 * morning + 1.3 * evening
        rate = (0.6 + 0.2 * station) * (0.2 + peaks) * np.where(dow >= 5, 0.6, 1.0)
        starts = hours.repeat(rng.poisson(rate))
        frames.append(
            pd.DataFrame(
                {
                    "Origen_Id": station,
                    "Destino_Id": rng.integers(1, N_STATIONS + 1, len(starts)),
                    "Inicio_del_viaje": starts
                    + pd.to_timedelta(rng.integers(0, 50, len(starts)), "m"),
                }
            )
        )
    trips = pd.concat(frames, ignore_index=True).sort_values("Inicio_del_viaje", kind="stable")
    trips = trips.reset_index(drop=True)
    n = len(trips)
    # De 3 a 39 minutos: ninguna regla de limpieza elimina estos viajes.
    trips["Fin_del_viaje"] = trips["Inicio_del_viaje"] + pd.to_timedelta(
        rng.integers(3, 40, n), "m"
    )
    trips["Viaje_Id"] = np.arange(1, n + 1)
    trips["Usuario_Id"] = rng.integers(1000, 2000, n)
    trips["Genero"] = rng.choice(["M", "F"], n)
    trips["Año_de_nacimiento"] = rng.integers(1960, 2005, n)
    return trips[list(TRIP_COLUMNS)]


def make_dirty_trips(valid: pd.DataFrame) -> pd.DataFrame:
    """Filas que la limpieza debe eliminar, una tanda por regla."""
    last = valid.tail(6).reset_index(drop=True)
    start = last["Inicio_del_viaje"]
    too_short = last.iloc[:3].assign(Fin_del_viaje=start[:3] + pd.Timedelta(seconds=30))
    too_long = last.iloc[3:4].assign(Fin_del_viaje=start[3:4] + pd.Timedelta(hours=5))
    unknown_station = last.iloc[4:6].assign(Destino_Id=999)
    dirty = pd.concat([too_short, too_long, unknown_station], ignore_index=True)
    dirty["Viaje_Id"] = valid["Viaje_Id"].max() + 1 + np.arange(len(dirty))
    duplicated = valid.tail(5)
    return pd.concat([dirty, duplicated], ignore_index=True)


def write_raw_files(raw_dir, seed: int = 0) -> dict[str, int]:
    """Escribe el catalogo y un CSV por mes. Devuelve cuantas filas validas y sucias hay."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    valid = make_raw_trips(seed)
    dirty = make_dirty_trips(valid)
    trips = pd.concat([valid, dirty], ignore_index=True)
    month = trips["Inicio_del_viaje"].dt.strftime("%Y_%m")
    # MiBici publica unos meses en UTF-8 y otros en Latin-1; se prueba una de cada una.
    for name, encoding in zip(MONTHS, ("utf-8", "latin-1")):
        path = raw_dir / f"datos_abiertos_{name}.csv"
        trips[month == name].to_csv(path, index=False, encoding=encoding)

    ids = np.arange(1, N_STATIONS + 1)
    stations = pd.DataFrame(
        {
            "id": ids,
            "name": [f"({'GDL' if i % 2 else 'ZPN'}-{i:03d}) Estación {i}" for i in ids],
            "latitude": 20.67 + ids * 0.002,
            "longitude": -103.35 - ids * 0.002,
            "dpcapacity": 10 + ids,
        }
    )
    stations.to_csv(raw_dir / STATIONS_FILE, index=False, encoding="latin-1")
    return {"valid": len(valid), "dirty": len(dirty)}


def build_panel_from_raw(raw_dir) -> SimpleNamespace:
    """Los mismos pasos de datos que mibici.pipeline, sin el EDA ni el versionado."""
    loader = MiBiciLoader(raw_dir)
    stations = loader.load_stations(STATIONS_FILE)
    raw = loader.load_trips(MONTHS)
    cleaner = TripCleaner(stations)
    clean = cleaner.clean(raw)
    config = FeatureConfig()
    # Como en el pipeline, los perfiles se ajustan solo con el periodo de entrenamiento.
    clusters = StationProfiler(config).fit_transform(
        clean[clean["start_time"] < pd.Timestamp(TEST_START)]
    )
    panel = DemandFeatureBuilder(stations, config).build(clean, clusters)
    return SimpleNamespace(
        stations=stations, raw=raw, clean=clean, cleaner=cleaner, clusters=clusters, panel=panel
    )


@pytest.fixture(scope="module")
def flow(tmp_path_factory):
    """Corre el flujo una sola vez; cada prueba revisa una etapa."""
    workdir = tmp_path_factory.mktemp("flujo")
    uri = f"sqlite:///{workdir / 'mlflow.db'}"
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(workdir)  # los artefactos de MLflow quedan en el directorio temporal
        counts = write_raw_files(workdir / "raw")
        data = build_panel_from_raw(workdir / "raw")

        validator = PanelValidator()
        validator.validate(data.panel)
        validator.raise_if_failed()

        experiment = DemandExperiment(
            data.panel,
            TrainingConfig(test_start=TEST_START, models=MODELS, n_iter=2, cv_splits=2),
            search_spaces=SEARCH_SPACES,
            tracking_uri=uri,
            experiment="integracion",
            registered_model=REGISTERED_MODEL,
            expected_fingerprint=None,
        )
        summary = experiment.run()

        # Inferencia como la haria el servicio: un objeto nuevo que solo conoce el registro.
        forecaster = DemandForecaster(model_name=REGISTERED_MODEL, tracking_uri=uri)
        _, test = temporal_split(data.panel, TEST_START)
        forecast = forecaster.predict(test)

        yield SimpleNamespace(
            workdir=workdir,
            uri=uri,
            counts=counts,
            validator=validator,
            experiment=experiment,
            summary=summary,
            forecaster=forecaster,
            test=test,
            forecast=forecast,
            **vars(data),
        )


def test_raw_files_are_loaded_and_cleaned(flow):
    assert len(flow.raw) == flow.counts["valid"] + flow.counts["dirty"]
    assert set(flow.raw["source_month"]) == set(MONTHS)
    assert set(TRIP_COLUMNS.values()) <= set(flow.raw.columns)
    assert flow.stations["zone"].isin(["GDL", "ZPN"]).all()
    assert flow.stations["station_name"].str.contains("Estación").all()  # acentos en Latin-1

    # La limpieza quita exactamente las filas sucias y deja registro de cada regla.
    assert len(flow.clean) == flow.counts["valid"]
    report = flow.cleaner.report_frame().set_index("regla")["filas_eliminadas"]
    assert report["duplicados_exactos"] == 5
    assert report["duracion_menor_1_min"] == 3
    assert report["duracion_mayor_180_min"] == 1
    assert report["estacion_fuera_de_catalogo"] == 2
    assert report.sum() == flow.counts["dirty"]


def test_panel_passes_validation_and_keeps_every_trip(flow):
    panel = flow.panel
    assert flow.validator.status == "ok"
    assert (flow.validator.report_frame()["resultado"] == "ok").all()
    assert panel["station_id"].nunique() == N_STATIONS
    assert len(flow.clusters) == N_STATIONS  # todas las estaciones tienen perfil
    assert len(panel) == N_STATIONS * panel["hour_ts"].nunique()

    # El panel conserva la demanda: sus salidas son los viajes limpios de esas mismas horas.
    in_panel = flow.clean["start_hour"] >= panel["hour_ts"].min()
    assert panel[TARGET].sum() == in_panel.sum()
    # El rezago de una semana es la demanda real de la misma estacion 168 horas antes.
    demand = panel.set_index(["station_id", "hour_ts"])[TARGET]
    late = panel[panel["hour_ts"] >= panel["hour_ts"].min() + pd.Timedelta(hours=168)]
    week_before = pd.MultiIndex.from_arrays(
        [late["station_id"], late["hour_ts"] - pd.Timedelta(hours=168)]
    )
    np.testing.assert_array_equal(late["dep_lag_168"], demand.loc[week_before])


def test_data_steps_are_reproducible(flow, tmp_path):
    # Mismos archivos de entrada y misma semilla: el panel debe ser identico bit a bit.
    again = build_panel_from_raw(flow.workdir / "raw")
    assert dataframe_fingerprint(again.panel) == dataframe_fingerprint(flow.panel)
    # Si cambian los datos crudos, la huella lo detecta.
    write_raw_files(tmp_path / "raw", seed=1)
    other = build_panel_from_raw(tmp_path / "raw")
    assert dataframe_fingerprint(other.panel) != dataframe_fingerprint(flow.panel)


def test_training_registers_the_models_with_their_lineage(flow):
    models = flow.summary[flow.summary["tipo"] == "modelo"]
    assert set(models["modelo"]) == set(MODELS)
    assert set(flow.summary.loc[flow.summary["tipo"] == "linea_base", "modelo"]) == set(
        NAIVE_BASELINES
    )
    champion = models[models["alias"] == CHAMPION_ALIAS].iloc[0]
    assert champion["cv_rmse"] == models["cv_rmse"].min()

    client = MlflowClient(tracking_uri=flow.uri)
    version = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS)
    assert int(version.version) == champion["version_registro"]
    assert version.run_id == champion["run_id"]
    # Linaje: la version registrada guarda la huella del panel con el que se entreno.
    assert version.tags["data.sha256"] == dataframe_fingerprint(flow.panel)
    assert version.tags["data.validacion"] == "ok"
    logged = client.get_run(champion["run_id"]).data.metrics
    assert logged["test_MAE"] == pytest.approx(champion["MAE"])


def test_forecast_from_the_registry_reproduces_the_logged_metrics(flow):
    forecast = flow.forecast
    assert len(forecast) == N_STATIONS * TEST_HOURS
    assert forecast["hour_ts"].min() == pd.Timestamp(TEST_START)
    assert not forecast.duplicated(["station_id", "hour_ts"]).any()
    assert (forecast[FORECAST_COLUMN] >= 0).all()
    assert (forecast["model_version"] == int(flow.forecaster.version.version)).all()

    # El modelo recargado del registro da el mismo pronostico que al entrenar...
    np.testing.assert_allclose(forecast[FORECAST_COLUMN], flow.experiment.best.test_pred)
    # ...y sus metricas son las que quedaron registradas en el run de MLflow.
    metrics = regression_metrics(forecast[TARGET], forecast[FORECAST_COLUMN])
    logged = MlflowClient(tracking_uri=flow.uri).get_run(flow.experiment.best.run_id)
    for name, value in metrics.items():
        assert logged.data.metrics[f"test_{name}"] == pytest.approx(value)

    # Criterio de exito del proyecto: el modelo supera a la regla ingenua de una semana antes.
    naive = regression_metrics(forecast[TARGET], flow.test[NAIVE_BASELINES[REFERENCE_BASELINE]])
    assert metrics["MAE"] < naive["MAE"]
    assert metrics["RMSE"] < naive["RMSE"]
