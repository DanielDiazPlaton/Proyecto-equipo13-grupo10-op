"""Particion temporal y metricas de evaluacion compartidas por todas las etapas.

La Fase 1 (lineas base) y la Fase 2 (modelos) usan estas mismas funciones, de modo
que los resultados son comparables entre si.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mibici.config import TEST_START

TIME_COLUMN = "hour_ts"

# Reglas fijas contra las que se compara todo modelo: nombre -> columna del panel
# que se usa directamente como pronostico.
NAIVE_BASELINES = {
    "ingenuo_lag_168h": "dep_lag_168",
    "media_movil_24h": "dep_roll_mean_24",
}


def temporal_split(
    panel: pd.DataFrame, test_start: str = TEST_START, time_col: str = TIME_COLUMN
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrenamiento antes de la fecha de corte y prueba despues (sin fuga de informacion)."""
    cut = pd.Timestamp(test_start)
    return panel[panel[time_col] < cut], panel[panel[time_col] >= cut]


def temporal_cv_splits(
    timestamps: pd.Series, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validacion cruzada de ventana expandible para un panel estacion-hora.

    El periodo se divide en n_splits + 1 bloques consecutivos de horas. En el
    pliegue i se entrena con los bloques 0..i-1 y se valida con el bloque i, por
    lo que la validacion siempre es posterior al entrenamiento. La particion se
    hace por hora y no por fila: todas las estaciones de una misma hora quedan
    del mismo lado.

    Devuelve indices posicionales, listos para el argumento cv de scikit-learn.
    """
    ts = pd.Series(timestamps).to_numpy()
    blocks = np.array_split(np.sort(np.unique(ts)), n_splits + 1)
    splits = []
    for block in blocks[1:]:
        train_idx = np.flatnonzero(ts < block[0])
        val_idx = np.flatnonzero((ts >= block[0]) & (ts <= block[-1]))
        splits.append((train_idx, val_idx))
    return splits


def clip_counts(pred) -> np.ndarray:
    """Un pronostico de viajes no puede ser negativo; los modelos lineales si pueden darlo."""
    return np.clip(np.asarray(pred, dtype=float), 0, None)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Metricas del pronostico de conteos.

    MAE: error promedio en viajes. RMSE: penaliza los errores grandes de las horas
    pico. WAPE: error absoluto total entre demanda total, comparable entre grupos
    de estaciones (MAPE no esta definida cuando hay ceros). Sesgo: promedio de
    pronostico menos real; negativo significa que el modelo subestima.
    """
    y = np.asarray(y_true, dtype=float)
    pred = clip_counts(y_pred)
    total = y.sum()
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "WAPE_pct": float(np.abs(y - pred).sum() / total * 100) if total > 0 else float("nan"),
        "sesgo": float((pred - y).mean()),
    }


def metrics_by_group(y_true, y_pred, groups, group_name: str = "grupo") -> pd.DataFrame:
    """Las mismas metricas calculadas por segmento (grupo de estacion, hora, municipio)."""
    frame = pd.DataFrame(
        {"y": np.asarray(y_true), "pred": np.asarray(y_pred), group_name: np.asarray(groups)}
    )
    rows = []
    for key, part in frame.groupby(group_name, sort=True):
        rows.append(
            {
                group_name: key,
                "filas": len(part),
                "demanda_media": float(part["y"].mean()),
                **regression_metrics(part["y"], part["pred"]),
            }
        )
    return pd.DataFrame(rows)


def baseline_metrics(test: pd.DataFrame, target: str) -> pd.DataFrame:
    """Evalua las reglas fijas (sin modelo) en el periodo de prueba."""
    rows = [
        {"modelo": name, **regression_metrics(test[target], test[column])}
        for name, column in NAIVE_BASELINES.items()
    ]
    return pd.DataFrame(rows)
