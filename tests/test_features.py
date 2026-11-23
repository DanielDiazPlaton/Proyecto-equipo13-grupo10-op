"""Pruebas unitarias del preprocesamiento: panel estacion-hora y transformador de variables."""

import numpy as np
import pandas as pd
import pytest

from mibici.features import (
    CATEGORICAL_FEATURES,
    COUNT_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    DemandFeatureBuilder,
    build_preprocessor,
)
from mibici.modeling.evaluation import temporal_split
from tests.conftest import PERIOD_START, TEST_START


def make_trips(rows: list[tuple]) -> pd.DataFrame:
    trips = pd.DataFrame(rows, columns=["origin_id", "destination_id", "start_time", "end_time"])
    trips["start_time"] = pd.to_datetime(trips["start_time"])
    trips["end_time"] = pd.to_datetime(trips["end_time"])
    trips["start_hour"] = trips["start_time"].dt.floor("h")
    return trips


def test_panel_counts_departures_and_arrivals_per_station_hour(stations):
    trips = make_trips(
        [
            (1, 2, "2026-09-01 08:05", "2026-09-01 08:20"),
            (1, 2, "2026-09-01 08:40", "2026-09-01 09:10"),  # llega en la hora siguiente
            (2, 1, "2026-09-01 10:15", "2026-09-01 10:30"),
        ]
    )
    panel = DemandFeatureBuilder(stations).build_panel(trips)
    cell = panel.set_index(["station_id", "hour_ts"])
    hour = pd.Timestamp("2026-09-01 08:00")

    # Rejilla completa: 2 estaciones x 3 horas, aunque la hora 9 no tenga salidas.
    assert len(panel) == 6
    assert cell.loc[(1, hour), "departures"] == 2
    assert cell.loc[(2, hour), "arrivals"] == 1
    assert cell.loc[(2, hour + pd.Timedelta(hours=1)), "arrivals"] == 1
    assert cell.loc[(1, hour + pd.Timedelta(hours=1)), "departures"] == 0
    assert panel["departures"].sum() == len(trips)
    assert (panel["net_flow"] == panel["arrivals"] - panel["departures"]).all()


def test_calendar_features():
    # Lunes a medianoche, miercoles 16 de septiembre (festivo) a las 6 h y sabado a las 18 h.
    ts = pd.to_datetime(["2026-09-14 00:00", "2026-09-16 06:00", "2026-09-19 18:00"])
    cal = DemandFeatureBuilder.add_calendar(pd.DataFrame({"hour_ts": ts}))

    assert cal["is_weekend"].tolist() == [0, 0, 1]
    assert cal["is_holiday"].tolist() == [0, 1, 0]
    # Codificacion ciclica: 0 h -> (0, 1), 6 h -> (1, 0), 18 h -> (-1, 0).
    assert cal["hour_sin"].to_numpy() == pytest.approx([0, 1, -1], abs=1e-12)
    assert cal["hour_cos"].to_numpy() == pytest.approx([1, 0, 0], abs=1e-12)
    assert cal.loc[0, ["dow_sin", "dow_cos"]].tolist() == pytest.approx([0, 1])
    assert (cal["hour_sin"] ** 2 + cal["hour_cos"] ** 2).to_numpy() == pytest.approx(1)


def test_lags_use_only_past_hours_of_the_same_station(stations):
    hours = pd.date_range("2026-09-01", periods=200, freq="h")
    frame = pd.concat(
        [
            pd.DataFrame({"station_id": s, "hour_ts": hours, "departures": np.arange(200) * s})
            for s in (1, 2)
        ],
        ignore_index=True,
    )
    # Las filas llegan desordenadas: el rezago debe salir del orden temporal, no del de llegada.
    out = DemandFeatureBuilder(stations).add_lags(frame.sample(frac=1, random_state=0))
    station = out[out["station_id"] == 2].set_index("hour_ts")
    row = station.loc[hours[180]]

    assert row["dep_lag_1"] == 179 * 2
    assert row["dep_lag_24"] == 156 * 2
    assert row["dep_lag_168"] == 12 * 2
    # El promedio movil cubre las 24 horas anteriores y no incluye la hora que se pronostica.
    assert row["dep_roll_mean_24"] == pytest.approx(np.arange(156, 180).mean() * 2)
    # La primera hora de cada estacion no hereda el historial de la otra estacion.
    assert out.loc[out["hour_ts"] == hours[0], "dep_lag_1"].isna().all()
    assert station["dep_lag_168"].isna().sum() == 168


def test_station_without_profile_gets_neutral_values(stations):
    panel = pd.DataFrame({"station_id": [1, 99], "hour_ts": pd.Timestamp("2026-09-01 08:00")})
    clusters = pd.DataFrame(
        {"station_id": [1], "pc1": [0.5], "pc2": [-0.5], "pc3": [0.1], "station_cluster": [2]}
    )
    out = DemandFeatureBuilder(stations).add_station_info(panel, clusters).set_index("station_id")

    assert out.loc[1, "station_cluster"] == "2"
    assert out.loc[1, "zone"] == "ZPN"
    # La estacion 99 no esta en el catalogo ni tiene perfil: grupo "-1", zona OTRA y PCA en 0.
    assert out.loc[99, "station_cluster"] == "-1"
    assert out.loc[99, "zone"] == "OTRA"
    assert out.loc[99, ["pc1", "pc2", "pc3"]].tolist() == [0.0, 0.0, 0.0]


def test_built_panel_has_model_columns_and_full_history(panel):
    assert set(MODEL_FEATURES + [TARGET]) <= set(panel.columns)
    assert panel[MODEL_FEATURES].notna().all().all()
    # Se descartan las primeras 168 horas de cada estacion, que no tienen el rezago semanal.
    assert panel["hour_ts"].min() == pd.Timestamp(PERIOD_START) + pd.Timedelta(hours=168)
    assert not panel.duplicated(["station_id", "hour_ts"]).any()


@pytest.mark.parametrize("log_counts", [False, True])
def test_preprocessor_imputes_scales_and_encodes(panel, log_counts):
    train, test = temporal_split(panel, TEST_START)
    features = train[MODEL_FEATURES].copy()
    features.loc[features.index[:50], ["dep_lag_24", "capacity"]] = np.nan

    preprocessor = build_preprocessor(log_counts=log_counts).fit(features)
    out = preprocessor.transform(features)
    n_numeric = len(NUMERIC_FEATURES)
    n_categories = sum(features[c].nunique() for c in CATEGORICAL_FEATURES)

    assert out.shape == (len(features), n_numeric + n_categories)
    assert np.isfinite(out).all()  # los nulos se imputaron
    assert out[:, :n_numeric].mean(axis=0) == pytest.approx(0, abs=1e-9)  # estandarizadas
    one_hot = out[:, n_numeric:]
    assert set(np.unique(one_hot)) <= {0.0, 1.0}
    assert (one_hot.sum(axis=1) == len(CATEGORICAL_FEATURES)).all()
    # La prueba se transforma con la media y la desviacion del entrenamiento, sin reajustar.
    assert np.isfinite(preprocessor.transform(test[MODEL_FEATURES])).all()


def test_log_counts_only_changes_count_features(panel):
    train, _ = temporal_split(panel, TEST_START)
    features = train[MODEL_FEATURES]
    plain = build_preprocessor(log_counts=False).fit_transform(features)
    logged = build_preprocessor(log_counts=True).fit_transform(features)
    n_counts = len(COUNT_FEATURES)  # los conteos son las primeras columnas de la salida

    assert not np.allclose(plain[:, :n_counts], logged[:, :n_counts])
    np.testing.assert_allclose(plain[:, n_counts:], logged[:, n_counts:])
