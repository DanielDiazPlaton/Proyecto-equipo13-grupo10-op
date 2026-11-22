"""Esquemas de entrada y salida de la API (Pydantic).

Cada instancia es una estacion en una hora, con la historia reciente de salidas de esa
estacion. El calendario y los atributos de la estacion no se piden: el servicio los
calcula con DemandFeatureBuilder y los toma de estaciones.csv.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_INSTANCES = 1000

# Estacion 51 (la de mas demanda) el jueves 24 de septiembre de 2026 a las 8 y a las
# 18 h, del periodo de prueba del panel v2.1; en las dos horas salieron 30 viajes.
# Sirve para probar POST /predict desde /docs.
PREDICT_EXAMPLE = {
    "instances": [
        {
            "station_id": 51,
            "hour_ts": "2026-09-24T08:00:00",
            "dep_lag_1": 32.0,
            "dep_lag_24": 31.0,
            "dep_lag_168": 25.0,
            "dep_roll_mean_24": 15.041666666666666,
        },
        {
            "station_id": 51,
            "hour_ts": "2026-09-24T18:00:00",
            "dep_lag_1": 31.0,
            "dep_lag_24": 39.0,
            "dep_lag_168": 41.0,
            "dep_roll_mean_24": 16.041666666666668,
        },
    ]
}


class Instance(BaseModel):
    """Una estacion en una hora, con sus salidas recientes."""

    station_id: int = Field(description="Identificador de la estacion en el catalogo de MiBici.")
    hour_ts: datetime = Field(
        description=(
            "Hora a pronosticar, en hora local de Guadalajara y sin zona horaria. "
            "Debe ser una hora exacta (minutos y segundos en cero)."
        )
    )
    dep_lag_1: float = Field(ge=0, allow_inf_nan=False, description="Salidas de la hora anterior.")
    dep_lag_24: float = Field(
        ge=0, allow_inf_nan=False, description="Salidas a la misma hora del dia anterior."
    )
    dep_lag_168: float = Field(
        ge=0, allow_inf_nan=False, description="Salidas a la misma hora de la semana anterior."
    )
    dep_roll_mean_24: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Promedio de salidas de las 24 horas anteriores (sin contar la hora actual).",
    )

    @field_validator("hour_ts")
    @classmethod
    def check_exact_hour(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            raise ValueError("hour_ts va en hora local de Guadalajara, sin zona horaria")
        if value.minute != 0 or value.second != 0 or value.microsecond != 0:
            raise ValueError("hour_ts debe ser una hora exacta, sin minutos ni segundos")
        return value


class PredictRequest(BaseModel):
    """Cuerpo de POST /predict."""

    model_config = ConfigDict(json_schema_extra={"examples": [PREDICT_EXAMPLE]})

    instances: list[Instance] = Field(
        min_length=1,
        max_length=MAX_INSTANCES,
        description=f"De 1 a {MAX_INSTANCES} combinaciones estacion-hora, sin repetirse.",
    )


class ModelRef(BaseModel):
    """Version del modelo que genero el pronostico."""

    name: str = Field(description="Nombre del modelo en el Model Registry.")
    version: int = Field(description="Version del modelo en el Model Registry.")
    model_uri: str = Field(description="URI del artefacto: models:/<nombre>/<version>.")


class Forecast(BaseModel):
    """Pronostico de una estacion en una hora."""

    station_id: int
    hour_ts: datetime
    pronostico: float = Field(ge=0, description="Viajes que se espera que inicien en esa hora.")


class PredictResponse(BaseModel):
    """Respuesta de POST /predict, en el mismo orden que las instancias recibidas."""

    model: ModelRef
    predictions: list[Forecast]


class HealthResponse(BaseModel):
    """Respuesta de GET /health."""

    status: str = Field(description="ok cuando el modelo esta cargado.")
    model_name: str
    model_version: int
