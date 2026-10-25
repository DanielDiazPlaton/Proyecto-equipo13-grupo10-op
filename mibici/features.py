"""Construccion del panel estacion-hora y preprocesamiento para el modelado."""

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mibici.config import HOLIDAYS, FeatureConfig


class StationProfiler:
    """Reduce el perfil horario de cada estacion con PCA y lo agrupa con KMeans.

    El perfil es la proporcion de salidas por hora (24 valores en dia habil y
    24 en fin de semana). Asi se distinguen estaciones de zonas de oficinas,
    habitacionales o de uso recreativo sin usar etiquetas externas.
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.pca = PCA(n_components=config.n_pca_components, random_state=config.random_state)
        self.kmeans = KMeans(
            n_clusters=config.k_clusters, n_init=10, random_state=config.random_state
        )
        self.silhouette_by_k: dict[int, float] = {}

    @staticmethod
    def build_profiles(trips: pd.DataFrame) -> pd.DataFrame:
        day_type = np.where(trips["start_time"].dt.dayofweek >= 5, "fds", "habil")
        prof = (
            trips.assign(day_type=day_type, hour=trips["start_time"].dt.hour)
            .groupby(["origin_id", "day_type", "hour"])
            .size()
            .unstack(["day_type", "hour"], fill_value=0)
        )
        prof.columns = [f"{d}_{h:02d}" for d, h in prof.columns]
        # El servicio no opera en algunas horas de madrugada; se completan con cero.
        habil = [f"habil_{h:02d}" for h in range(24)]
        fds = [f"fds_{h:02d}" for h in range(24)]
        prof = prof.reindex(columns=habil + fds, fill_value=0)
        prof[habil] = prof[habil].div(prof[habil].sum(axis=1).replace(0, 1), axis=0)
        prof[fds] = prof[fds].div(prof[fds].sum(axis=1).replace(0, 1), axis=0)
        return prof

    def fit_transform(self, trips: pd.DataFrame, min_trips: int = 100) -> pd.DataFrame:
        counts = trips["origin_id"].value_counts()
        active = counts[counts >= min_trips].index
        profiles = self.build_profiles(trips[trips["origin_id"].isin(active)])
        self.profiles_ = profiles
        # Los perfiles ya son proporciones (misma escala), por eso no se estandarizan:
        # estandarizar daria el mismo peso a las horas de madrugada, que son casi ruido.
        self.full_variance_ = PCA(n_components=10).fit(profiles).explained_variance_ratio_
        comps = self.pca.fit_transform(profiles)
        for k in range(2, 8):
            labels = KMeans(
                n_clusters=k, n_init=10, random_state=self.config.random_state
            ).fit_predict(comps)
            self.silhouette_by_k[k] = float(silhouette_score(comps, labels))
        labels = self.kmeans.fit_predict(comps)
        out = pd.DataFrame(
            comps, index=profiles.index, columns=[f"pc{i + 1}" for i in range(comps.shape[1])]
        )
        out["station_cluster"] = labels
        out.index.name = "station_id"
        logger.info(f"Varianza explicada PCA: {self.pca.explained_variance_ratio_.round(3)}")
        return out.reset_index()


class DemandFeatureBuilder:
    """Convierte viajes en un panel estacion-hora con la variable objetivo y sus features."""

    def __init__(self, stations: pd.DataFrame, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.stations = stations

    def build_panel(self, trips: pd.DataFrame) -> pd.DataFrame:
        hours = pd.date_range(trips["start_hour"].min(), trips["start_hour"].max(), freq="h")
        station_ids = np.sort(trips["origin_id"].unique())
        grid = pd.MultiIndex.from_product([station_ids, hours], names=["station_id", "hour_ts"])

        departures = trips.groupby(["origin_id", "start_hour"]).size()
        departures.index.names = ["station_id", "hour_ts"]
        end_hour = trips["end_time"].dt.floor("h")
        arrivals = trips.groupby([trips["destination_id"], end_hour]).size()
        arrivals.index.names = ["station_id", "hour_ts"]

        panel = pd.DataFrame(index=grid)
        panel["departures"] = departures.reindex(grid, fill_value=0).astype(np.int32)
        panel["arrivals"] = arrivals.reindex(grid, fill_value=0).astype(np.int32)
        panel["net_flow"] = panel["arrivals"] - panel["departures"]
        return panel.reset_index()

    @staticmethod
    def add_calendar(panel: pd.DataFrame) -> pd.DataFrame:
        ts = panel["hour_ts"]
        panel["hour"] = ts.dt.hour.astype(np.int8)
        panel["dayofweek"] = ts.dt.dayofweek.astype(np.int8)
        panel["is_weekend"] = (panel["dayofweek"] >= 5).astype(np.int8)
        panel["is_holiday"] = ts.dt.strftime("%Y-%m-%d").isin(HOLIDAYS).astype(np.int8)
        # Codificacion ciclica: la hora 23 queda cerca de la hora 0.
        panel["hour_sin"] = np.sin(2 * np.pi * panel["hour"] / 24)
        panel["hour_cos"] = np.cos(2 * np.pi * panel["hour"] / 24)
        panel["dow_sin"] = np.sin(2 * np.pi * panel["dayofweek"] / 7)
        panel["dow_cos"] = np.cos(2 * np.pi * panel["dayofweek"] / 7)
        return panel

    def add_lags(self, panel: pd.DataFrame) -> pd.DataFrame:
        panel = panel.sort_values(["station_id", "hour_ts"])
        grouped = panel.groupby("station_id")["departures"]
        for lag in self.config.lags:
            panel[f"dep_lag_{lag}"] = grouped.shift(lag)
        w = self.config.rolling_window
        panel[f"dep_roll_mean_{w}"] = grouped.transform(lambda s: s.shift(1).rolling(w).mean())
        return panel

    def add_station_info(self, panel: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
        info = self.stations[["station_id", "lat", "lon", "capacity", "zone"]]
        panel = panel.merge(info, on="station_id", how="left")
        panel = panel.merge(clusters, on="station_id", how="left")
        # Estaciones con menos de 100 viajes no tienen perfil confiable.
        pcs = [c for c in clusters.columns if c.startswith("pc")]
        panel[pcs] = panel[pcs].fillna(0.0)
        panel["station_cluster"] = panel["station_cluster"].fillna(-1).astype(int).astype(str)
        panel["zone"] = panel["zone"].fillna("OTRA")
        return panel

    def build(self, trips: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
        panel = self.build_panel(trips)
        panel = self.add_calendar(panel)
        panel = self.add_lags(panel)
        panel = self.add_station_info(panel, clusters)
        # Las primeras 168 horas de cada estacion no tienen historia completa.
        before = len(panel)
        panel = panel.dropna(subset=[f"dep_lag_{max(self.config.lags)}"]).reset_index(drop=True)
        logger.info(f"Panel: {len(panel):,} filas (se quitaron {before - len(panel):,} sin lags)")
        return panel


NUMERIC_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_holiday",
    "dep_lag_1",
    "dep_lag_24",
    "dep_lag_168",
    "dep_roll_mean_24",
    "capacity",
    "lat",
    "lon",
    "pc1",
    "pc2",
    "pc3",
]
CATEGORICAL_FEATURES = ["zone", "station_cluster"]
TARGET = "departures"


def build_preprocessor() -> ColumnTransformer:
    """Estandariza numericas y aplica one-hot a categoricas."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
