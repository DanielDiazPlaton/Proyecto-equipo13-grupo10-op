"""Entrenamiento, ajuste, evaluacion y registro de los modelos de demanda (Fase 2).

Flujo:
    1. Carga el panel estacion-hora (v2) y lo parte por fecha en entrenamiento y prueba.
    2. Arma un Pipeline de scikit-learn por modelo candidato (preprocesamiento + modelo).
    3. Ajusta hiperparametros con busqueda aleatoria y validacion cruzada temporal.
    4. Evalua el mejor pipeline de cada modelo en el periodo de prueba.
    5. Registra todo en MLflow: un run por modelo y un run anidado por combinacion probada.
    6. Registra los pipelines en el Model Registry y asigna el alias champion al mejor.

Uso:
    python -m mibici.modeling.train                         # todos los modelos
    python -m mibici.modeling.train --model ridge --n-iter 4
    python -m mibici.modeling.train --no-register           # solo experimentos
"""

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Annotated

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import typer

from mibici.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MLFLOW_MODEL_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    PANEL_FILE,
    PANEL_VERSION,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    SEARCH_SPACES,
    SKOPS_TRUSTED_TYPES,
    TABLES_DIR,
    TrainingConfig,
)
from mibici.features import MODEL_FEATURES, build_preprocessor
from mibici.modeling.evaluation import (
    NAIVE_BASELINES,
    TIME_COLUMN,
    baseline_metrics,
    clip_counts,
    metrics_by_group,
    regression_metrics,
    temporal_cv_splits,
    temporal_split,
)
from mibici.versioning import dataframe_fingerprint

app = typer.Typer()

MODEL_STEP = "model"
REFERENCE_BASELINE = "ingenuo_lag_168h"


@dataclass(frozen=True)
class ModelSpec:
    """Un modelo candidato: el estimador, su preprocesamiento y por que se eligio."""

    name: str
    estimator: BaseEstimator
    log_counts: bool
    rationale: str


def model_catalog(random_state: int = RANDOM_STATE) -> dict[str, ModelSpec]:
    """Modelos candidatos, del mas simple al mas flexible."""
    specs = [
        ModelSpec(
            "ridge",
            Ridge(),
            log_counts=False,
            rationale=(
                "Referencia lineal con error cuadratico y regularizacion L2. Es el modelo de "
                "la Fase 1, ahora con alpha ajustado; mide cuanto aporta cada modelo mas complejo."
            ),
        ),
        ModelSpec(
            "poisson_glm",
            PoissonRegressor(solver="newton-cholesky", max_iter=300),
            log_counts=True,
            rationale=(
                "Modelo lineal generalizado con distribucion Poisson y enlace log. Adecuado para "
                "conteos: el pronostico nunca es negativo y la varianza crece con la media."
            ),
        ),
        ModelSpec(
            "gradient_boosting",
            HistGradientBoostingRegressor(early_stopping=False, random_state=random_state),
            log_counts=False,
            rationale=(
                "Gradient boosting por histogramas con perdida Poisson o cuadratica (se ajusta). "
                "Captura interacciones no lineales, por ejemplo entre la hora y el tipo de "
                "estacion, y escala a cientos de miles de filas."
            ),
        ),
    ]
    return {spec.name: spec for spec in specs}


def build_model_pipeline(spec: ModelSpec) -> Pipeline:
    """Pipeline completo: del panel con sus columnas originales al pronostico.

    El preprocesamiento va dentro del pipeline, asi que en cada pliegue de la
    validacion cruzada se ajusta solo con los datos de entrenamiento de ese pliegue
    y el modelo registrado recibe los datos sin transformar.
    """
    return Pipeline(
        [
            ("preprocess", build_preprocessor(log_counts=spec.log_counts)),
            (MODEL_STEP, clone(spec.estimator)),
        ]
    )


def _rmse(y_true, y_pred) -> float:
    return regression_metrics(y_true, y_pred)["RMSE"]


def _mae(y_true, y_pred) -> float:
    return regression_metrics(y_true, y_pred)["MAE"]


# Las mismas metricas de la evaluacion final, como scorers de scikit-learn.
SCORERS = {
    "rmse": make_scorer(_rmse, greater_is_better=False),
    "mae": make_scorer(_mae, greater_is_better=False),
}


def plain_params(params: dict) -> dict:
    """Quita el prefijo del paso del Pipeline: model__alpha -> alpha."""
    return {key.split("__", 1)[-1]: value for key, value in params.items()}


@dataclass
class ModelResult:
    """Resultado del mejor pipeline de un modelo candidato."""

    name: str
    run_id: str
    model_uri: str
    pipeline: Pipeline
    best_params: dict
    cv_scores: dict
    test_metrics: dict
    test_pred: np.ndarray


class DemandExperiment:
    """Ajusta, evalua y registra en MLflow los modelos candidatos sobre un panel."""

    def __init__(
        self,
        panel: pd.DataFrame,
        config: TrainingConfig | None = None,
        search_spaces: dict[str, dict[str, list]] | None = None,
        tracking_uri: str = MLFLOW_TRACKING_URI,
        experiment: str = MLFLOW_MODEL_EXPERIMENT,
        registered_model: str = REGISTERED_MODEL_NAME,
        data_source: str = str(PANEL_FILE),
    ):
        self.config = config or TrainingConfig()
        self.search_spaces = search_spaces or SEARCH_SPACES
        self.registered_model = registered_model
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.client = MlflowClient()

        train, test = temporal_split(panel, self.config.test_start)
        if train.empty or test.empty:
            raise ValueError(f"La fecha de corte {self.config.test_start} deja un periodo vacio")
        self.train = train.reset_index(drop=True)
        self.test = test.reset_index(drop=True)
        self.cv = temporal_cv_splits(self.train[TIME_COLUMN], self.config.cv_splits)

        # Linaje: cada run queda ligado a la version del panel con la que se entreno.
        self.data_tags = {
            "data.name": "mibici_panel",
            "data.version": PANEL_VERSION,
            "data.sha256": dataframe_fingerprint(panel),
        }
        self.dataset = mlflow.data.from_pandas(
            panel.head(1000), source=data_source, name="mibici_panel"
        )
        self.results: list[ModelResult] = []
        self.trials: list[dict] = []
        logger.info(f"Entrenamiento: {len(self.train):,} filas. Prueba: {len(self.test):,} filas")

    @property
    def target(self) -> str:
        return self.config.target

    @property
    def selection_key(self) -> str:
        return f"cv_{self.config.selection_metric}"

    # Lineas base
    def log_baselines(self) -> pd.DataFrame:
        """Registra las reglas fijas como runs para compararlas con los modelos en MLflow."""
        table = baseline_metrics(self.test, self.target)
        for row in table.to_dict("records"):
            name = row.pop("modelo")
            with mlflow.start_run(run_name=name):
                mlflow.set_tags({"tipo": "linea_base", "modelo": name, **self.data_tags})
                mlflow.log_params(
                    {
                        "test_start": self.config.test_start,
                        "columna_pronostico": NAIVE_BASELINES[name],
                    }
                )
                mlflow.log_metrics({f"test_{k}": v for k, v in row.items()})
        return table

    # Ajuste de hiperparametros
    def tune(self, spec: ModelSpec) -> RandomizedSearchCV:
        """Busqueda aleatoria con validacion cruzada temporal sobre el Pipeline completo."""
        space = self.search_spaces[spec.name]
        search = RandomizedSearchCV(
            build_model_pipeline(spec),
            space,
            n_iter=min(self.config.n_iter, len(ParameterGrid(space))),
            scoring=SCORERS,
            refit=self.config.selection_metric,
            cv=self.cv,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
        )
        search.fit(self.train[MODEL_FEATURES], self.train[self.target])
        return search

    def _log_trials(self, spec: ModelSpec, search: RandomizedSearchCV) -> None:
        """Un run anidado por combinacion: permite compararlas en la interfaz de MLflow."""
        cv = search.cv_results_
        for i, params in enumerate(cv["params"]):
            metrics = {
                "cv_rmse": float(-cv["mean_test_rmse"][i]),
                "cv_rmse_std": float(cv["std_test_rmse"][i]),
                "cv_mae": float(-cv["mean_test_mae"][i]),
                "tiempo_ajuste_s": float(cv["mean_fit_time"][i]),
            }
            with mlflow.start_run(run_name=f"{spec.name}-combinacion-{i + 1:02d}", nested=True):
                mlflow.set_tags({"tipo": "combinacion", "modelo": spec.name, **self.data_tags})
                mlflow.log_params(plain_params(params))
                mlflow.log_metrics(metrics)
            self.trials.append(
                {
                    "modelo": spec.name,
                    "combinacion": i + 1,
                    "es_mejor": i == search.best_index_,
                    **metrics,
                    "hiperparametros": json.dumps(plain_params(params)),
                }
            )

    # Entrenamiento y evaluacion de un modelo
    def run_model(self, spec: ModelSpec) -> ModelResult:
        """Ajusta un modelo, lo evalua en prueba y deja todo en un run de MLflow."""
        cfg = self.config
        x_test, y_test = self.test[MODEL_FEATURES], self.test[self.target]
        logger.info(f"Ajustando {spec.name}")
        with mlflow.start_run(run_name=spec.name) as run:
            mlflow.set_tags(
                {
                    "tipo": "modelo",
                    "modelo": spec.name,
                    "mlflow.note.content": spec.rationale,
                    **self.data_tags,
                }
            )
            mlflow.log_input(self.dataset, context="training")
            run_params = {k: v for k, v in cfg.to_params().items() if k != "models"}
            run_params.update(
                {
                    "n_train": len(self.train),
                    "n_test": len(self.test),
                    "n_features": len(MODEL_FEATURES),
                    "log_counts": spec.log_counts,
                }
            )
            mlflow.log_params(run_params)

            search = self.tune(spec)
            self._log_trials(spec, search)
            best = search.best_estimator_
            best_params = plain_params(search.best_params_)
            mlflow.log_params(best_params)

            cv = search.cv_results_
            cv_scores = {
                "cv_rmse": float(-cv["mean_test_rmse"][search.best_index_]),
                "cv_mae": float(-cv["mean_test_mae"][search.best_index_]),
            }
            pred = clip_counts(best.predict(x_test))
            test_metrics = regression_metrics(y_test, pred)
            mlflow.log_metrics({**cv_scores, **{f"test_{k}": v for k, v in test_metrics.items()}})

            by_cluster = metrics_by_group(y_test, pred, self.test["station_cluster"])
            for row in by_cluster.itertuples(index=False):
                mlflow.log_metric(f"test_MAE_grupo_{row.grupo}", row.MAE)
                mlflow.log_metric(f"test_WAPE_pct_grupo_{row.grupo}", row.WAPE_pct)
            mlflow.log_text(by_cluster.to_csv(index=False), "metricas_por_grupo.csv")
            mlflow.log_text(pd.DataFrame(cv).to_csv(index=False), "busqueda_hiperparametros.csv")

            sample = self.train[MODEL_FEATURES].head(200)
            model_info = mlflow.sklearn.log_model(
                best,
                name="model",
                signature=infer_signature(sample, best.predict(sample)),
                input_example=sample.head(5),
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
        logger.info(
            f"{spec.name}: cv_rmse={cv_scores['cv_rmse']:.4f} "
            f"test_MAE={test_metrics['MAE']:.4f} test_RMSE={test_metrics['RMSE']:.4f}"
        )
        return ModelResult(
            name=spec.name,
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
            pipeline=best,
            best_params=best_params,
            cv_scores=cv_scores,
            test_metrics=test_metrics,
            test_pred=pred,
        )

    # Model Registry
    def _current_champion_score(self) -> float:
        """Metrica de validacion del champion actual; infinito si todavia no hay uno."""
        try:
            version = self.client.get_model_version_by_alias(self.registered_model, CHAMPION_ALIAS)
            return float(version.tags[self.selection_key])
        except (mlflow.exceptions.MlflowException, KeyError, ValueError):
            return float("inf")

    def register(self, results: list[ModelResult]) -> dict[str, dict]:
        """Registra cada pipeline como una version y actualiza los alias.

        El orden se decide con la metrica de validacion cruzada, no con la de prueba,
        para que el periodo de prueba siga siendo una estimacion honesta del error.
        El mejor de esta corrida toma el alias champion solo si iguala o mejora al
        champion vigente; el segundo queda como challenger.
        """
        ranked = sorted(results, key=lambda r: r.cv_scores[self.selection_key])
        champion_score = self._current_champion_score()
        registry = {}
        for position, res in enumerate(ranked):
            tags = {
                "modelo": res.name,
                "hiperparametros": json.dumps(res.best_params),
                **{k: f"{v:.6f}" for k, v in res.cv_scores.items()},
                **{f"test_{k}": f"{v:.6f}" for k, v in res.test_metrics.items()},
                **self.data_tags,
            }
            version = mlflow.register_model(res.model_uri, self.registered_model, tags=tags)
            self.client.update_model_version(
                self.registered_model,
                version.version,
                description=(
                    f"{res.name} ajustado con validacion temporal. "
                    f"Prueba desde {self.config.test_start}: MAE {res.test_metrics['MAE']:.3f}, "
                    f"RMSE {res.test_metrics['RMSE']:.3f}, WAPE {res.test_metrics['WAPE_pct']:.1f} %."
                ),
            )
            alias = ""
            if position == 0 and res.cv_scores[self.selection_key] <= champion_score:
                alias = CHAMPION_ALIAS
            elif position == 1:
                alias = CHALLENGER_ALIAS
            if alias:
                self.client.set_registered_model_alias(
                    self.registered_model, alias, version.version
                )
            registry[res.name] = {"version_registro": int(version.version), "alias": alias}
        self.client.update_registered_model(
            self.registered_model,
            description=(
                "Pronostico de viajes que inician por estacion y hora en MiBici. Cada version "
                "es un Pipeline de scikit-learn (preprocesamiento + modelo) que recibe las "
                "columnas del panel estacion-hora."
            ),
        )
        return registry

    # Orquestacion
    def run(self, register: bool = True) -> pd.DataFrame:
        """Corre lineas base y modelos, y devuelve la tabla comparativa."""
        catalog = model_catalog(self.config.random_state)
        baselines = self.log_baselines()
        self.results = [self.run_model(catalog[name]) for name in self.config.models]
        registry = self.register(self.results) if register else {}
        return self.summary(baselines, registry)

    def summary(self, baselines: pd.DataFrame, registry: dict[str, dict]) -> pd.DataFrame:
        base = baselines.assign(tipo="linea_base")
        ranked = sorted(self.results, key=lambda r: r.cv_scores[self.selection_key])
        models = pd.DataFrame(
            [
                {
                    "modelo": r.name,
                    "tipo": "modelo",
                    **r.cv_scores,
                    **r.test_metrics,
                    **registry.get(r.name, {}),
                    "run_id": r.run_id,
                }
                for r in ranked
            ]
        )
        table = pd.concat([base, models], ignore_index=True)
        reference = table.loc[table["modelo"] == REFERENCE_BASELINE, "MAE"].iloc[0]
        table["mejora_MAE_vs_ingenuo_pct"] = (1 - table["MAE"] / reference) * 100
        if "version_registro" in table.columns:
            table["version_registro"] = table["version_registro"].astype("Int64")
        first = ["modelo", "tipo", "cv_rmse", "cv_mae", "MAE", "RMSE", "WAPE_pct", "sesgo"]
        return table[first + [c for c in table.columns if c not in first]]

    @property
    def best(self) -> ModelResult:
        return min(self.results, key=lambda r: r.cv_scores[self.selection_key])

    def test_frame(self) -> pd.DataFrame:
        """Periodo de prueba con el pronostico del mejor modelo y de la regla de referencia."""
        frame = self.test[["station_id", TIME_COLUMN, "hour", "station_cluster", self.target]]
        return frame.assign(
            pronostico=self.best.test_pred,
            ingenuo=self.test[NAIVE_BASELINES[REFERENCE_BASELINE]].to_numpy(),
        )

    def segment_metrics(self) -> pd.DataFrame:
        """Mejor modelo contra la regla de referencia, por grupo de estacion."""
        frame = self.test_frame()
        parts = []
        for label, column in [(self.best.name, "pronostico"), (REFERENCE_BASELINE, "ingenuo")]:
            part = metrics_by_group(frame[self.target], frame[column], frame["station_cluster"])
            parts.append(part.assign(modelo=label))
        return pd.concat(parts, ignore_index=True)

    def feature_importance(self) -> pd.DataFrame:
        """Importancia por permutacion del mejor modelo sobre una muestra de prueba.

        Mide cuanto sube el RMSE al desordenar una variable. Se calcula sobre las
        columnas originales del panel, no sobre las transformadas.
        """
        n = min(self.config.importance_sample, len(self.test))
        sample = self.test.sample(n=n, random_state=self.config.random_state)
        result = permutation_importance(
            self.best.pipeline,
            sample[MODEL_FEATURES],
            sample[self.target],
            scoring=SCORERS["rmse"],
            n_repeats=3,
            random_state=self.config.random_state,
        )
        table = pd.DataFrame(
            {
                "variable": MODEL_FEATURES,
                "aumento_rmse": result.importances_mean,
                "desviacion": result.importances_std,
            }
        )
        return table.sort_values("aumento_rmse", ascending=False).reset_index(drop=True)


def save_reports(experiment: DemandExperiment, summary: pd.DataFrame, tables_dir: Path) -> None:
    """Deja en reports/ las tablas y figuras que respaldan los resultados."""
    from mibici.plots import ModelReport

    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "resultados_modelos.csv", index=False)
    trials = pd.DataFrame(experiment.trials)
    trials.to_csv(tables_dir / "busqueda_hiperparametros.csv", index=False)
    segments = experiment.segment_metrics()
    segments.to_csv(tables_dir / "metricas_por_grupo.csv", index=False)
    importance = experiment.feature_importance()
    importance.to_csv(tables_dir / "importancia_variables.csv", index=False)
    best_params = {r.name: r.best_params for r in experiment.results}
    (tables_dir / "mejores_hiperparametros.json").write_text(json.dumps(best_params, indent=2))

    report = ModelReport()
    report.plot_model_comparison(summary)
    report.plot_search(trials)
    report.plot_forecast(experiment.test_frame(), experiment.target, experiment.best.name)
    report.plot_error_by_hour(experiment.test_frame(), experiment.target, experiment.best.name)
    report.plot_importance(importance, experiment.best.name)
    with mlflow.start_run(run_id=experiment.best.run_id):
        for path in report.files:
            mlflow.log_artifact(str(path), artifact_path="figuras")


@app.command()
def main(
    panel_path: Path = PANEL_FILE,
    model: Annotated[
        list[str] | None, typer.Option(help="Modelo a entrenar; se puede repetir.")
    ] = None,
    n_iter: Annotated[
        int | None, typer.Option(help="Combinaciones de hiperparametros por modelo.")
    ] = None,
    register: Annotated[
        bool, typer.Option(help="Registrar los modelos en el Model Registry.")
    ] = True,
    report: Annotated[bool, typer.Option(help="Generar tablas y figuras en reports/.")] = True,
):
    if not Path(panel_path).exists():
        raise typer.BadParameter(f"No existe {panel_path}. Corre primero: make data")
    config = TrainingConfig()
    if model:
        unknown = set(model) - set(model_catalog())
        if unknown:
            raise typer.BadParameter(f"Modelos desconocidos: {sorted(unknown)}")
        config = replace(config, models=tuple(model))
    if n_iter:
        config = replace(config, n_iter=n_iter)

    panel = pd.read_parquet(panel_path)
    experiment = DemandExperiment(panel, config, data_source=str(panel_path))
    summary = experiment.run(register=register)
    if report:
        save_reports(experiment, summary, TABLES_DIR)
    columns = ["modelo", "cv_rmse", "MAE", "RMSE", "WAPE_pct", "mejora_MAE_vs_ingenuo_pct"]
    logger.info("\n" + summary[columns].round(3).to_string(index=False))
    logger.success(f"Mejor modelo: {experiment.best.name}")


if __name__ == "__main__":
    app()
