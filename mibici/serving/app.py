"""API de pronostico de demanda MiBici (FastAPI).

Endpoints:
    GET  /health   estado del servicio y version del modelo
    GET  /model    metadata del modelo servido (model_info.json)
    POST /predict  pronostico de salidas por estacion y hora

El modelo se carga una sola vez al arrancar, desde la carpeta de la variable de entorno
MODEL_DIR (por defecto models/mibici-demanda-salidas). Si no se puede cargar, la API
arranca de todos modos y responde 503 hasta que se corrija.

Uso:
    uvicorn mibici.serving.app:app --port 8000      # make api
    Documentacion interactiva en http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from loguru import logger
from mlflow.exceptions import MlflowException

from mibici.config import SERVING_MODEL_DIR
from mibici.modeling.predict import FORECAST_COLUMN
from mibici.serving.schemas import (
    MAX_INSTANCES,
    Forecast,
    HealthResponse,
    ModelRef,
    PredictRequest,
    PredictResponse,
)
from mibici.serving.service import ForecastService, UnknownStationError
from mibici.validation import ERROR, DataValidationError

API_VERSION = "1.0.0"

DESCRIPTION = f"""
Pronostico de los viajes que inician en cada estacion de MiBici (Guadalajara) en una hora.

Cada instancia trae la estacion, la hora y las salidas recientes de esa estacion
(`dep_lag_1`, `dep_lag_24`, `dep_lag_168` y `dep_roll_mean_24`). El servicio calcula el
calendario y agrega los atributos fijos de la estacion antes de llamar al modelo.

Se aceptan de 1 a {MAX_INSTANCES} instancias por peticion. Errores:

- **404**: alguna estacion no existe en el catalogo del modelo.
- **422**: la entrada no cumple el esquema, o el panel no pasa la validacion de datos
  (por ejemplo, una estacion-hora repetida).
- **503**: el modelo no esta cargado.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.getenv("MODEL_DIR", str(SERVING_MODEL_DIR)))
    app.state.service = None
    try:
        app.state.service = ForecastService(model_dir)
    except (OSError, ValueError, KeyError, MlflowException):
        # Sin modelo la API sigue arriba: /health responde 503 y el HEALTHCHECK lo detecta.
        logger.exception(f"No se pudo cargar el modelo desde {model_dir}")
    yield
    app.state.service = None


app = FastAPI(
    title="MiBici: pronostico de demanda por estacion",
    description=DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)


def get_service(request: Request) -> ForecastService:
    """Servicio cargado al arrancar; 503 si el modelo no se pudo cargar."""
    service = request.app.state.service
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no esta cargado. Revisa MODEL_DIR y los logs del servicio.",
        )
    return service


LoadedService = Annotated[ForecastService, Depends(get_service)]
NOT_LOADED = {503: {"description": "El modelo no esta cargado"}}


@app.get("/health", response_model=HealthResponse, responses=NOT_LOADED)
def health(service: LoadedService) -> HealthResponse:
    """Estado del servicio y version del modelo cargado."""
    return HealthResponse(
        status="ok", model_name=service.model_name, model_version=service.version
    )


@app.get("/model", responses=NOT_LOADED)
def model_info(service: LoadedService) -> dict:
    """Metadata del modelo servido: version, model_uri, hiperparametros, metricas y datos."""
    return service.info


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        404: {"description": "Alguna estacion no existe en estaciones.csv"},
        422: {
            "description": (
                "La entrada no cumple el esquema (Pydantic) o el panel no pasa la "
                "validacion de PanelValidator"
            )
        },
        **NOT_LOADED,
    },
)
def predict(body: PredictRequest, service: LoadedService) -> PredictResponse:
    """Pronostico de salidas por estacion y hora, en el orden de las instancias."""
    try:
        forecast = service.predict(body.instances)
    except UnknownStationError as error:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "estacion_desconocida",
                "mensaje": str(error),
                "station_ids": error.station_ids,
            },
        )
    except DataValidationError as error:
        rules = []
        for row in error.report.itertuples(index=False):
            if row.resultado == ERROR:
                rules.append({"regla": row.regla, "nota": row.nota})
        raise HTTPException(
            status_code=422,
            detail={
                "error": "validacion_panel",
                "mensaje": "El panel armado con las instancias no paso la validacion",
                "reglas": rules,
            },
        )

    predictions = []
    for row in forecast.itertuples(index=False):
        predictions.append(
            Forecast(
                station_id=row.station_id,
                hour_ts=row.hour_ts,
                pronostico=getattr(row, FORECAST_COLUMN),
            )
        )
    return PredictResponse(
        model=ModelRef(
            name=service.model_name, version=service.version, model_uri=service.model_uri
        ),
        predictions=predictions,
    )
