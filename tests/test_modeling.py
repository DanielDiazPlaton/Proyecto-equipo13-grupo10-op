"""Pruebas del pipeline de modelado: particion temporal, metricas, pipelines y registro."""

import numpy as np
import pandas as pd
import pytest

from mibici.config import CHAMPION_ALIAS, TrainingConfig
from mibici.features import MODEL_FEATURES, TARGET
from mibici.modeling.evaluation import (
    regression_metrics,
    temporal_cv_splits,
    temporal_split,
)
from mibici.modeling.predict import FORECAST_COLUMN, DemandForecaster
from mibici.modeling.train import DemandExperiment, build_model_pipeline, model_catalog
from tests.conftest import TEST_START

SMALL_SPACES = {
    "ridge": {"model__alpha": [0.1, 10.0]},
    "poisson_glm": {"model__alpha": [1e-4, 1e-2]},
    "gradient_boosting": {"model__loss": ["poisson"], "model__max_iter": [20, 40]},
}


def test_temporal_split_does_not_mix_periods(panel):
    train, test = temporal_split(panel, TEST_START)
    assert len(train) + len(test) == len(panel)
    assert train["hour_ts"].max() < pd.Timestamp(TEST_START) <= test["hour_ts"].min()


def test_cv_validates_only_on_later_hours(panel):
    train, _ = temporal_split(panel, TEST_START)
    ts = train["hour_ts"].reset_index(drop=True)
    splits = temporal_cv_splits(ts, n_splits=3)
    assert len(splits) == 3
    previous_size = 0
    for train_idx, val_idx in splits:
        assert ts[train_idx].max() < ts[val_idx].min()
        assert len(train_idx) > previous_size  # ventana expandible
        previous_size = len(train_idx)


def test_metrics_known_values_and_clipping():
    metrics = regression_metrics([0, 2, 4], [1, 2, -3])  # -3 se recorta a 0
    assert metrics["MAE"] == pytest.approx(5 / 3)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(17 / 3))
    assert metrics["WAPE_pct"] == pytest.approx(5 / 6 * 100)
    assert metrics["sesgo"] == pytest.approx(-1.0)


@pytest.mark.parametrize("name", ["ridge", "poisson_glm", "gradient_boosting"])
def test_pipeline_fits_on_raw_panel_columns(panel, name):
    train, test = temporal_split(panel, TEST_START)
    pipeline = build_model_pipeline(model_catalog()[name])
    pipeline.fit(train[MODEL_FEATURES], train[TARGET])
    pred = pipeline.predict(test[MODEL_FEATURES])
    assert pred.shape == (len(test),)
    assert np.isfinite(pred).all()
    if name == "poisson_glm":  # el enlace log garantiza pronosticos no negativos
        assert (pred >= 0).all()


def test_pipeline_handles_unseen_category(panel):
    train, test = temporal_split(panel, TEST_START)
    pipeline = build_model_pipeline(model_catalog()["ridge"])
    pipeline.fit(train[MODEL_FEATURES], train[TARGET])
    new_station = test[MODEL_FEATURES].head(5).assign(zone="OTRA", station_cluster="-1")
    assert np.isfinite(pipeline.predict(new_station)).all()


def test_experiment_registers_champion_and_forecaster_loads_it(panel, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # los artefactos de MLflow quedan en el directorio temporal
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config = TrainingConfig(test_start=TEST_START, n_iter=2, cv_splits=2)
    experiment = DemandExperiment(
        panel,
        config,
        search_spaces=SMALL_SPACES,
        tracking_uri=uri,
        experiment="prueba",
        registered_model="modelo-prueba",
    )
    summary = experiment.run()

    models = summary[summary["tipo"] == "modelo"]
    assert set(models["modelo"]) == set(config.models)
    assert models["version_registro"].notna().all()
    champion = models[models["alias"] == CHAMPION_ALIAS]
    assert len(champion) == 1
    assert champion["cv_rmse"].iloc[0] == models["cv_rmse"].min()
    # Con demanda sintetica de patron diario, el modelo debe superar a la regla ingenua.
    naive_mae = summary.loc[summary["modelo"] == "ingenuo_lag_168h", "MAE"].iloc[0]
    assert champion["MAE"].iloc[0] < naive_mae

    forecaster = DemandForecaster(model_name="modelo-prueba", tracking_uri=uri)
    assert forecaster.version.tags["modelo"] == champion["modelo"].iloc[0]
    forecast = forecaster.predict(experiment.test)
    assert len(forecast) == len(experiment.test)
    assert (forecast[FORECAST_COLUMN] >= 0).all()
    np.testing.assert_allclose(forecast[FORECAST_COLUMN], experiment.best.test_pred)
