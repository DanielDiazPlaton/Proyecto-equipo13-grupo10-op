"""Pruebas de la API de pronostico (FastAPI) con TestClient.

No usan el modelo real: entrenan un Ridge chico con el panel sintetico de conftest.py y lo
guardan como lo deja mibici.modeling.export (model/, estaciones.csv y model_info.json) en
un directorio temporal al que apunta MODEL_DIR.
"""

import json

from fastapi.testclient import TestClient
import mlflow
import numpy as np
import pytest

from mibici.config import (
    SERVING_INFO_FILE,
    SERVING_MODEL_SUBDIR,
    SERVING_STATIONS_FILE,
    SKOPS_TRUSTED_TYPES,
)
from mibici.features import MODEL_FEATURES, TARGET
from mibici.modeling.evaluation import clip_counts
from mibici.modeling.export import station_attributes
from mibici.modeling.train import build_model_pipeline, model_catalog
from mibici.serving.app import app

from .conftest import TEST_START

LAG_COLUMNS = ["dep_lag_1", "dep_lag_24", "dep_lag_168", "dep_roll_mean_24"]


@pytest.fixture(scope="module")
def model_dir(panel, tmp_path_factory):
    """Carpeta con un modelo exportado de prueba."""
    folder = tmp_path_factory.mktemp("modelo_exportado")
    train = panel[panel["hour_ts"] < TEST_START]
    pipeline = build_model_pipeline(model_catalog()["ridge"])
    pipeline.fit(train[MODEL_FEATURES], train[TARGET])
    mlflow.sklearn.save_model(
        pipeline, str(folder / SERVING_MODEL_SUBDIR), skops_trusted_types=SKOPS_TRUSTED_TYPES
    )
    station_attributes(panel).to_csv(folder / SERVING_STATIONS_FILE, index=False)
    info = {
        "registered_model_name": "modelo-de-prueba",
        "version": 3,
        "model_uri": "models:/modelo-de-prueba/3",
        "algoritmo": "ridge",
    }
    (folder / SERVING_INFO_FILE).write_text(json.dumps(info), encoding="utf-8")
    return folder, pipeline


@pytest.fixture(scope="module")
def client(model_dir):
    folder, _ = model_dir
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MODEL_DIR", str(folder))
        with TestClient(app) as test_client:  # el with corre el arranque (lifespan)
            yield test_client


def sample_test_rows(panel, n: int = 5):
    rows = panel[panel["hour_ts"] >= TEST_START].sample(n=n, random_state=0)
    return rows.reset_index(drop=True)


def make_instance(row) -> dict:
    instance = {"station_id": int(row["station_id"]), "hour_ts": row["hour_ts"].isoformat()}
    for column in LAG_COLUMNS:
        instance[column] = float(row[column])
    return instance


def test_health_reports_the_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_name": "modelo-de-prueba",
        "model_version": 3,
    }


def test_predict_returns_one_non_negative_forecast_per_instance(client, model_dir, panel):
    _, pipeline = model_dir
    rows = sample_test_rows(panel)
    instances = []
    for _, row in rows.iterrows():
        instances.append(make_instance(row))

    response = client.post("/predict", json={"instances": instances})

    assert response.status_code == 200
    body = response.json()
    assert body["model"]["model_uri"] == "models:/modelo-de-prueba/3"
    assert len(body["predictions"]) == len(instances)
    forecasts = []
    for instance, prediction in zip(instances, body["predictions"]):
        assert prediction["station_id"] == instance["station_id"]
        assert prediction["pronostico"] >= 0
        forecasts.append(prediction["pronostico"])
    # El servicio arma las mismas variables que el panel: mismo pronostico que el pipeline.
    expected = clip_counts(pipeline.predict(rows[MODEL_FEATURES]))
    np.testing.assert_allclose(forecasts, expected, rtol=0, atol=1e-9)


def test_unknown_station_returns_404(client, panel):
    instance = make_instance(sample_test_rows(panel, n=1).iloc[0])
    instance["station_id"] = 99999

    response = client.post("/predict", json={"instances": [instance]})

    assert response.status_code == 404
    assert response.json()["detail"]["station_ids"] == [99999]


@pytest.mark.parametrize(
    "change",
    [
        {"hour_ts": "2026-09-24T08:30:00"},  # no es una hora exacta
        {"dep_lag_1": -1},  # rezago negativo
        {"station_id": "estacion"},  # no es un entero
    ],
)
def test_invalid_input_returns_422(client, panel, change):
    instance = make_instance(sample_test_rows(panel, n=1).iloc[0])
    instance.update(change)
    response = client.post("/predict", json={"instances": [instance]})
    assert response.status_code == 422


def test_repeated_station_hour_fails_the_panel_validation(client, panel):
    instance = make_instance(sample_test_rows(panel, n=1).iloc[0])

    response = client.post("/predict", json={"instances": [instance, instance]})

    assert response.status_code == 422
    rules = response.json()["detail"]["reglas"]
    assert rules[0]["regla"] == "llave_sin_duplicados"


def test_missing_model_returns_503(tmp_path, monkeypatch, panel):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "no_existe"))
    instance = make_instance(sample_test_rows(panel, n=1).iloc[0])
    with TestClient(app) as test_client:
        assert test_client.get("/health").status_code == 503
        assert test_client.post("/predict", json={"instances": [instance]}).status_code == 503
