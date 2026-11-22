"""Pruebas unitarias de la inferencia: DemandForecaster.predict con un modelo simulado.

El modelo real se carga del Model Registry (eso lo cubre tests/test_integration.py). Aqui se
sustituye por un doble para probar solo la logica propia del pronostico, sin MLflow.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from mibici.features import MODEL_FEATURES, TARGET
from mibici.modeling.predict import FORECAST_COLUMN, DemandForecaster
from mibici.validation import DataValidationError


class FakeModel:
    """Devuelve valores de -1 a 3 y guarda las columnas que recibio."""

    def __init__(self):
        self.received_columns = None

    def predict(self, features):
        self.received_columns = list(features.columns)
        return np.linspace(-1, 3, len(features))


def make_forecaster(model: FakeModel, version: str = "7") -> DemandForecaster:
    forecaster = DemandForecaster.__new__(DemandForecaster)  # sin conectar al Model Registry
    forecaster.model_name, forecaster.alias = "modelo-simulado", "champion"
    forecaster.version = SimpleNamespace(version=version, tags={})
    forecaster.model = model
    return forecaster


def test_forecast_keeps_keys_and_clips_negative_values(panel):
    model = FakeModel()
    forecast = make_forecaster(model).predict(panel)

    assert list(forecast.columns) == [
        "station_id",
        "hour_ts",
        TARGET,
        FORECAST_COLUMN,
        "model_version",
    ]
    assert len(forecast) == len(panel)
    assert (forecast["station_id"].to_numpy() == panel["station_id"].to_numpy()).all()
    assert forecast[FORECAST_COLUMN].min() == 0  # el -1 del modelo se recorta a 0
    assert forecast[FORECAST_COLUMN].max() == pytest.approx(3)
    assert (forecast["model_version"] == 7).all()


def test_model_receives_only_the_model_features_in_order(panel):
    model = FakeModel()
    make_forecaster(model).predict(panel)
    assert model.received_columns == MODEL_FEATURES


def test_forecast_works_without_the_real_value(panel):
    # En produccion el panel de las horas futuras no trae la variable objetivo.
    forecast = make_forecaster(FakeModel()).predict(panel.drop(columns=[TARGET]))
    assert TARGET not in forecast.columns
    assert len(forecast) == len(panel)


def test_invalid_panel_is_rejected_before_calling_the_model(panel):
    model = FakeModel()
    broken = panel.drop(columns=["dep_lag_24"])
    with pytest.raises(DataValidationError):
        make_forecaster(model).predict(broken)
    assert model.received_columns is None  # el modelo nunca se llamo
