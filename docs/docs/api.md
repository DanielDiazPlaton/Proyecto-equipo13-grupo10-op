# API del modelo

El modelo `champion` se sirve con FastAPI. El código está en `mibici/serving/` y el modelo
que se carga es la carpeta `models/mibici-demanda-salidas/`, que deja `make export-model`:

| Archivo | Contenido |
|---|---|
| `model/` | El pipeline de scikit-learn en formato MLflow (`MLmodel` y `model.skops`) |
| `estaciones.csv` | Atributos fijos de las 469 estaciones: `lat`, `lon`, `capacity`, `zone`, `pc1`, `pc2`, `pc3` y `station_cluster` |
| `model_info.json` | Nombre, versión, alias, `model_uri`, run, hiperparámetros, métricas, versión y huella de los datos, versiones de las librerías |

El modelo servido es `models:/mibici-demanda-salidas/1` (alias `champion`): gradient boosting
por histogramas con MAE de prueba 0.862 y RMSE de prueba 1.502. La API no necesita
`mlflow.db` ni el Model Registry.

## Cómo correrla

```bash
make api                     # local, en http://localhost:8000
make docker-build            # docker build -t ml-service:latest .
make docker-run              # docker run -p 8000:8000 ml-service:latest
docker run -p 8000:8000 luisadr18/mibici-demanda:1.0.0   # imagen publicada en Docker Hub
```

La carpeta del modelo se puede cambiar con la variable de entorno `MODEL_DIR`.

## Endpoints

| Método y ruta | Qué devuelve |
|---|---|
| `GET /health` | `status`, nombre y versión del modelo cargado |
| `GET /model` | El contenido de `model_info.json` |
| `POST /predict` | El pronóstico de salidas por estación y hora |
| `GET /docs` | Documentación interactiva (Swagger UI), con un ejemplo listo para probar |
| `GET /openapi.json` | El esquema OpenAPI |

## Entrada de `POST /predict`

Un objeto con `instances`, una lista de 1 a 1,000 combinaciones estación-hora sin repetirse.
El servicio calcula el calendario (hora, día, fin de semana, festivo y codificación cíclica)
con `DemandFeatureBuilder.add_calendar()` y agrega los atributos de la estación desde
`estaciones.csv`, así que solo se piden estos campos:

| Campo | Tipo | Regla |
|---|---|---|
| `station_id` | entero | Debe existir en `estaciones.csv` |
| `hour_ts` | fecha y hora | Hora local de Guadalajara, sin zona horaria y sin minutos ni segundos |
| `dep_lag_1` | número | ≥ 0. Salidas de la hora anterior |
| `dep_lag_24` | número | ≥ 0. Salidas a la misma hora del día anterior |
| `dep_lag_168` | número | ≥ 0. Salidas a la misma hora de la semana anterior |
| `dep_roll_mean_24` | número | ≥ 0. Promedio de salidas de las 24 horas anteriores |

## Salida

```json
{
  "model": {"name": "mibici-demanda-salidas", "version": 1, "model_uri": "models:/mibici-demanda-salidas/1"},
  "predictions": [
    {"station_id": 51, "hour_ts": "2026-09-24T08:00:00", "pronostico": 30.720685542405807},
    {"station_id": 51, "hour_ts": "2026-09-24T18:00:00", "pronostico": 34.50356440913216}
  ]
}
```

Es la respuesta al ejemplo de `/docs` (estación 51, jueves 24 de septiembre a las 8 y a las
18 h; en las dos horas salieron 30 viajes). Las predicciones salen en el mismo orden que las
instancias. `pronostico` son los viajes que se espera que inicien en esa estación y hora;
nunca es negativo (`clip_counts`).

## Errores

| Código | Cuándo | Cuerpo |
|---|---|---|
| 422 | La entrada no cumple el esquema: falta un campo, `hour_ts` con minutos, un rezago negativo, más de 1,000 instancias | El detalle estándar de Pydantic, campo por campo |
| 404 | Alguna estación no existe en `estaciones.csv` | `{"detail": {"error": "estacion_desconocida", "mensaje": "...", "station_ids": [99999]}}` |
| 422 | El panel armado con las instancias no pasa `PanelValidator(require_target=False)`, por ejemplo una estación-hora repetida | `{"detail": {"error": "validacion_panel", "mensaje": "...", "reglas": [{"regla": "llave_sin_duplicados", "nota": "..."}]}}` |
| 503 | El modelo no se pudo cargar al arrancar (por ejemplo, `MODEL_DIR` apunta a una carpeta vacía) | `{"detail": "El modelo no esta cargado. ..."}` |

## Esquema OpenAPI

FastAPI genera el esquema a partir de los modelos de Pydantic de `mibici/serving/schemas.py`.
Además de `/docs` y `/openapi.json` con la API corriendo, se guarda una copia en
`docs/openapi.json`:

```bash
make openapi
```

Hay que volver a generarla cada vez que cambien los endpoints o los esquemas.
