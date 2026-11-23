"""Pruebas unitarias del calculo de metricas: totales, por segmento y lineas base."""

import numpy as np
import pytest

from mibici.features import TARGET
from mibici.modeling.evaluation import (
    NAIVE_BASELINES,
    baseline_metrics,
    clip_counts,
    metrics_by_group,
    regression_metrics,
    temporal_split,
)
from tests.conftest import TEST_START


def test_clip_counts_removes_negative_forecasts():
    clipped = clip_counts([-2, 0.5, 3])
    assert clipped.tolist() == [0.0, 0.5, 3.0]
    assert clipped.dtype == float


def test_perfect_forecast_has_zero_error():
    metrics = regression_metrics([0, 3, 7], [0, 3, 7])
    assert metrics == {"MAE": 0.0, "RMSE": 0.0, "WAPE_pct": 0.0, "sesgo": 0.0}


def test_bias_sign_shows_under_and_over_forecast():
    assert regression_metrics([4, 6], [3, 5])["sesgo"] == pytest.approx(-1.0)  # subestima
    assert regression_metrics([4, 6], [6, 8])["sesgo"] == pytest.approx(2.0)  # sobreestima


def test_wape_is_undefined_without_demand():
    # Una estacion sin viajes en el periodo no tiene demanda contra la cual medir el error.
    metrics = regression_metrics([0, 0], [1, 0])
    assert np.isnan(metrics["WAPE_pct"])
    assert metrics["MAE"] == pytest.approx(0.5)


def test_metrics_by_group_match_each_segment():
    y = [2, 4, 0, 10, 6]
    pred = [1, 4, 1, 7, 6]
    groups = ["a", "a", "b", "b", "b"]
    table = metrics_by_group(y, pred, groups).set_index("grupo")

    assert table["filas"].to_dict() == {"a": 2, "b": 3}
    assert table.loc["b", "demanda_media"] == pytest.approx(16 / 3)
    for group, rows in {"a": slice(0, 2), "b": slice(2, 5)}.items():
        expected = regression_metrics(y[rows], pred[rows])
        assert table.loc[group, list(expected)].to_dict() == pytest.approx(expected)


def test_baseline_metrics_use_the_panel_columns(panel):
    _, test = temporal_split(panel, TEST_START)
    table = baseline_metrics(test, TARGET).set_index("modelo")

    assert list(table.index) == list(NAIVE_BASELINES)
    for name, column in NAIVE_BASELINES.items():
        expected = regression_metrics(test[TARGET], test[column])
        assert table.loc[name].to_dict() == pytest.approx(expected)
