"""Datos sinteticos compartidos por las pruebas. Son pequenos para que corran en segundos."""

import numpy as np
import pandas as pd
import pytest

from mibici.config import FeatureConfig
from mibici.features import DemandFeatureBuilder

N_STATIONS = 6
PERIOD_START = "2026-09-01"
PERIOD_END = "2026-09-28 23:00"
TEST_START = "2026-09-22"


def make_synthetic_trips(seed: int = 0) -> pd.DataFrame:
    """Viajes con dos picos diarios (8 h y 18 h) y menos demanda en fin de semana."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(PERIOD_START, PERIOD_END, freq="h")
    hour, dow = hours.hour.to_numpy(), hours.day_of_week.to_numpy()
    shape = 0.2 + np.exp(-((hour - 8) ** 2) / 8) + 1.3 * np.exp(-((hour - 18) ** 2) / 8)
    frames = []
    for station in range(1, N_STATIONS + 1):
        rate = station * 0.8 * shape * np.where(dow >= 5, 0.6, 1.0)
        starts = hours.repeat(rng.poisson(rate))
        frames.append(
            pd.DataFrame(
                {
                    "origin_id": station,
                    "destination_id": rng.integers(1, N_STATIONS + 1, len(starts)),
                    "start_time": starts + pd.to_timedelta(rng.integers(0, 50, len(starts)), "m"),
                }
            )
        )
    trips = pd.concat(frames, ignore_index=True)
    trips["end_time"] = trips["start_time"] + pd.Timedelta(minutes=9)
    trips["start_hour"] = trips["start_time"].dt.floor("h")
    return trips


@pytest.fixture(scope="session")
def stations() -> pd.DataFrame:
    ids = np.arange(1, N_STATIONS + 1)
    return pd.DataFrame(
        {
            "station_id": ids,
            "lat": 20.67 + ids * 0.002,
            "lon": -103.35 - ids * 0.002,
            "capacity": 10 + ids,
            "zone": np.where(ids % 2 == 0, "GDL", "ZPN"),
        }
    )


@pytest.fixture(scope="session")
def panel(stations) -> pd.DataFrame:
    """Panel estacion-hora construido con el mismo DemandFeatureBuilder del proyecto."""
    ids = stations["station_id"]
    clusters = pd.DataFrame(
        {
            "station_id": ids,
            "pc1": np.linspace(-1, 1, len(ids)),
            "pc2": np.linspace(1, -1, len(ids)),
            "pc3": 0.0,
            "station_cluster": ids % 3,
        }
    )
    return DemandFeatureBuilder(stations, FeatureConfig()).build(make_synthetic_trips(), clusters)
