"""Pipeline de datos (Fase 1): carga, limpieza, EDA, features y versionado en MLflow.

Uso:
    python -m mibici.pipeline            # corre todo
    python -m mibici.pipeline --skip-eda # solo datos y versiones

El entrenamiento de modelos (Fase 2) esta en mibici/modeling/train.py y parte del
panel que deja este pipeline.
"""

import json

from loguru import logger
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import typer

from mibici.cleaning import TripCleaner, iqr_upper_fence
from mibici.config import (
    CLEAN_TRIPS_FILE,
    EXPECTED_FINGERPRINTS,
    MONTHS,
    PANEL_FILE,
    PANEL_VERSION,
    TABLES_DIR,
    TEST_START,
    CleaningConfig,
    FeatureConfig,
)
from mibici.dataset import MiBiciLoader
from mibici.features import (
    MODEL_FEATURES,
    TARGET,
    DemandFeatureBuilder,
    StationProfiler,
    build_preprocessor,
)
from mibici.modeling.evaluation import baseline_metrics, regression_metrics, temporal_split
from mibici.plots import EDAReport
from mibici.validation import PANEL_REPORT_FILE, PanelValidator
from mibici.versioning import DatasetVersioner

app = typer.Typer()


def evaluate_baselines(panel: pd.DataFrame, test_start: str = TEST_START) -> pd.DataFrame:
    """Lineas base para validar que el panel tiene senal predictiva. No es el modelo final.

    Compara las dos reglas fijas con una regresion Ridge sin ajustar. El ajuste de
    hiperparametros y la comparacion de modelos se hacen en mibici/modeling/train.py.
    """
    train, test = temporal_split(panel, test_start)
    ridge = Pipeline([("preprocess", build_preprocessor()), ("model", Ridge(alpha=1.0))])
    ridge.fit(train[MODEL_FEATURES], train[TARGET])
    ridge_row = {
        "modelo": "ridge_features_escaladas",
        **regression_metrics(test[TARGET], ridge.predict(test[MODEL_FEATURES])),
    }
    return pd.concat(
        [baseline_metrics(test, TARGET), pd.DataFrame([ridge_row])], ignore_index=True
    )


@app.command()
def main(skip_eda: bool = False):
    loader = MiBiciLoader()
    versioner = DatasetVersioner()
    CLEAN_TRIPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    tables_dir = TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)

    # v0: datos crudos tal como se descargan
    stations = loader.load_stations()
    trips = loader.load_trips(MONTHS)
    raw_duration = (
        pd.to_datetime(trips["end_time"]) - pd.to_datetime(trips["start_time"])
    ).dt.total_seconds() / 60
    versioner.log_version(
        trips,
        "mibici_viajes",
        "v0",
        "raw",
        params={"meses": ",".join(MONTHS)},
        metrics={
            "nulos_genero": int(trips["gender"].isna().sum()),
            "nulos_anio_nacimiento": int(trips["birth_year"].isna().sum()),
            "duplicados_exactos": int(trips.drop(columns="source_month").duplicated().sum()),
            "viajes_menor_1_min": int((raw_duration < 1).sum()),
        },
    )

    # v1: viajes limpios (regla de duracion maxima por dominio)
    clean_cfg = CleaningConfig()
    cleaner = TripCleaner(stations, clean_cfg)
    clean = cleaner.clean(trips)
    report = cleaner.report_frame()
    report_path = tables_dir / "reporte_limpieza.csv"
    report.to_csv(report_path, index=False)
    clean_path = CLEAN_TRIPS_FILE
    clean.to_parquet(clean_path, index=False)
    versioner.log_version(
        clean,
        "mibici_viajes",
        "v1",
        "interim",
        params=clean_cfg.to_params(),
        metrics={
            "filas_eliminadas": len(trips) - len(clean),
            "pct_conservado": len(clean) / len(trips) * 100,
        },
        artifacts=[report_path],
        file_path=clean_path,
        parent_version="v0",
    )

    # v1b: variante experimental con limite de Tukey sobre log(duracion)
    fence = iqr_upper_fence(clean["duration_min"])
    alt_cfg = CleaningConfig(max_duration_min=round(fence, 1))
    alt = TripCleaner(stations, alt_cfg).clean(trips)
    versioner.log_version(
        alt,
        "mibici_viajes",
        "v1b",
        "interim",
        params=alt_cfg.to_params(),
        metrics={
            "filas_eliminadas": len(trips) - len(alt),
            "pct_conservado": len(alt) / len(trips) * 100,
        },
        parent_version="v0",
    )

    # Panel estacion-hora con features para modelado (entrada de la Fase 2)
    feat_cfg = FeatureConfig()
    profiler = StationProfiler(feat_cfg)
    # Los perfiles se ajustan solo con el periodo de entrenamiento para no filtrar
    # informacion del periodo de prueba hacia las features.
    clusters = profiler.fit_transform(clean[clean["start_time"] < pd.Timestamp(TEST_START)])
    panel = DemandFeatureBuilder(stations, feat_cfg).build(clean, clusters)
    # El panel no se guarda si no cumple el esquema que espera el modelado. El reporte
    # se guarda antes de detener el proceso para poder revisar que regla fallo.
    validator = PanelValidator()
    validator.validate(panel, expected_fingerprint=EXPECTED_FINGERPRINTS.get(PANEL_VERSION))
    validation_path = validator.save_report(PANEL_REPORT_FILE)
    validator.raise_if_failed()
    panel_path = PANEL_FILE
    panel.to_parquet(panel_path, index=False)
    baselines = evaluate_baselines(panel)
    baselines_path = tables_dir / "lineas_base.csv"
    baselines.to_csv(baselines_path, index=False)
    metrics = {
        "pca_varianza_3c": float(profiler.pca.explained_variance_ratio_.sum()),
        "silueta_k": profiler.silhouette_by_k[feat_cfg.k_clusters],
        "pct_horas_sin_viajes": float((panel[TARGET] == 0).mean() * 100),
    }
    metrics.update({f"MAE_{r.modelo}": r.MAE for r in baselines.itertuples()})
    versioner.log_version(
        panel,
        "mibici_panel",
        PANEL_VERSION,
        "processed",
        params=feat_cfg.to_params(),
        metrics=metrics,
        artifacts=[baselines_path, validation_path],
        file_path=panel_path,
        parent_version="v1",
    )

    history = versioner.history()
    history.to_csv(tables_dir / "historial_versiones.csv", index=False)

    if not skip_eda:
        eda = EDAReport(clean, stations)
        summary = eda.summary()
        (tables_dir / "resumen_eda.json").write_text(json.dumps(summary, indent=2, default=str))
        eda.duration_table().to_csv(tables_dir / "duracion_descriptiva.csv")
        eda.plot_daily()
        eda.plot_hourly_profile()
        eda.plot_duration(raw_duration)
        eda.plot_heatmap()
        eda.plot_top_stations()
        eda.plot_demographics()
        eda.plot_net_flow()
        eda.plot_station_map(clusters)
        eda.plot_cluster_profiles(profiler.profiles_, clusters)
        corr_cols = [
            TARGET,
            "arrivals",
            "dep_lag_1",
            "dep_lag_24",
            "dep_lag_168",
            "dep_roll_mean_24",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "capacity",
            "pc1",
            "pc2",
            "pc3",
        ]
        eda.plot_correlation(panel, corr_cols)
        eda.plot_pca_variance(profiler.full_variance_, profiler.silhouette_by_k)
        extra = {
            "silueta_por_k": profiler.silhouette_by_k,
            "pca_varianza": profiler.pca.explained_variance_ratio_.tolist(),
            "limite_tukey_min": fence,
            "filas_v1b": len(alt),
            "panel_filas": len(panel),
            "panel_columnas": panel.shape[1],
        }
        (tables_dir / "extra.json").write_text(json.dumps(extra, indent=2, default=str))
    logger.success("Pipeline de la Fase 1 terminado")


if __name__ == "__main__":
    app()
