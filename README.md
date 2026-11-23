# mibici-mlops

Proyecto del curso Operaciones de aprendizaje automático (Gpo 10), Tecnológico de Monterrey.

Equipo 13:

| Integrante | Rol principal |
|---|---|
| Daniel Díaz Platón | Dueño de producto / analista de negocio |
| Luis Ángel Díaz Rebollo | Ingeniero de datos |
| Manuel Díaz Borrego | Científico de datos |
| José Carlos Cruz Martínez | Ingeniero de ML / MLOps y DevOps |

Objetivo: pronosticar la demanda horaria de bicicletas por estación del sistema MiBici
(Área Metropolitana de Guadalajara) para apoyar la redistribución de bicicletas.

## Cómo reproducir

Se requiere Python 3.11 o superior.

```bash
make create_environment        # crea .venv
source .venv/bin/activate      # en Windows: .venv\Scripts\activate
make requirements
python -c "from mibici.dataset import MiBiciLoader as L; l = L(); l.download_trips(2026, [7, 8, 9]); l.download_stations()"
make data       # Fase 1: limpieza, features, EDA y versiones del dataset en MLflow
make train      # Fase 2: ajuste, evaluación y registro de modelos en MLflow
make predict    # pronóstico con el modelo champion del Model Registry
make test       # todas las pruebas; equivale a: pytest -q
make mlflow-ui  # interfaz de MLflow en http://localhost:5000
make export-model  # Fase 3: exporta el champion a models/ para servirlo
make api           # API de pronostico en http://localhost:8000/docs
```

Datos usados: viajes de julio, agosto y septiembre de 2026 y el catálogo de estaciones
`nomenclatura_2026_10.csv`, publicados en https://mibici.net/es/datos-abiertos/.
Los cuatro CSV viven en `data/raw/` y no se suben al repositorio por su tamaño. No hace falta
descargarlos a mano: `MiBiciLoader.download_trips()` y `MiBiciLoader.download_stations()`
localizan los enlaces en el portal y guardan los archivos en `data/raw/`; si un archivo ya
existe, se omite la descarga. El notebook `1.0-equipo13-eda-fase1.ipynb` muestra el uso.

## Fase 1: datos

Carga, limpieza, análisis exploratorio, construcción de variables y registro de cada versión
del dataset en MLflow (experimento `mibici-fase1-datos`).

| Versión | Etapa | Descripción |
|---|---|---|
| v0 | raw | Viajes crudos concatenados de los tres meses |
| v1 | interim | Viajes limpios, duración máxima de 180 min (criterio de dominio) |
| v1b | interim | Variante experimental, duración máxima por regla de Tukey sobre log(duración) |
| v2 | processed | Panel estación-hora con variables de calendario, rezagos y perfil de estación (catálogo de septiembre) |
| v2.1 | processed | El mismo panel con los atributos de estación del catálogo de octubre; es la versión vigente |

Cada run guarda parámetros de limpieza, métricas de filas eliminadas, esquema, huella
SHA-256 del contenido y el archivo parquet resultante.

## Fase 2: modelos

`make train` corre `mibici/modeling/train.py` sobre el panel v2.1:

1. **Partición temporal.** Entrenamiento hasta el 16 de septiembre y prueba del 17 al 30.
2. **Pipeline de scikit-learn.** Cada modelo es un `Pipeline` con dos pasos: `preprocess`
   (imputación, escalado, `log1p` opcional de los rezagos y one-hot) y `model`. El
   preprocesamiento se ajusta solo con los datos de entrenamiento de cada pliegue.
3. **Modelos candidatos.**

   | Modelo | Por qué |
   |---|---|
   | `ridge` | Referencia lineal de la Fase 1, ahora con `alpha` ajustado |
   | `poisson_glm` | Regresión Poisson: modelo para conteos, pronóstico no negativo |
   | `gradient_boosting` | `HistGradientBoostingRegressor` con pérdida Poisson o cuadrática; captura interacciones no lineales |

4. **Ajuste de hiperparámetros.** Búsqueda aleatoria (`RandomizedSearchCV`) con validación
   cruzada temporal de ventana expandible. Los espacios de búsqueda están en
   `mibici/config.py` (`SEARCH_SPACES`).
5. **Evaluación.** MAE, RMSE, WAPE y sesgo en el periodo de prueba, en total y por grupo de
   estación, comparados contra la regla "misma hora de la semana anterior".
6. **Seguimiento en MLflow** (experimento `mibici-fasefinal-modelos`): un run por modelo con
   hiperparámetros, métricas, tablas, figuras y el pipeline; un run anidado por cada
   combinación probada; y un run por cada línea base. Todos llevan la versión y la huella
   SHA-256 del panel con el que se entrenaron.
7. **Model Registry.** Cada pipeline queda como una versión del modelo
   `mibici-demanda-salidas`, con sus hiperparámetros y métricas como etiquetas. El mejor en
   validación recibe el alias `champion` y el segundo `challenger`.

`make predict` carga la versión `champion` y escribe el pronóstico en
`data/processed/pronostico_salidas.parquet`.

Opciones útiles:

```bash
python -m mibici.modeling.train --model ridge --model poisson_glm   # solo algunos modelos
python -m mibici.modeling.train --n-iter 4 --no-register            # prueba rápida
python -m mibici.modeling.predict --alias challenger
```

Los resultados quedan en `reports/tables/` (`resultados_modelos.csv`,
`busqueda_hiperparametros.csv`, `metricas_por_grupo.csv`, `importancia_variables.csv`,
`mejores_hiperparametros.json`) y en `reports/figures/` (figuras 14 a 18).

## Pruebas

Todas las pruebas se corren con un solo comando desde la raíz del repositorio:

```bash
pytest -q
```

`make test` hace lo mismo. Las pruebas usan datos sintéticos y un `mlflow.db` temporal, así que
no necesitan los CSV de MiBici ni haber corrido `make data` o `make train`, y no tocan los
experimentos del proyecto. Son 52 pruebas y tardan alrededor de un minuto y medio.

| Archivo | Tipo | Qué valida |
|---|---|---|
| `tests/test_data.py` | Unitarias | Reglas de limpieza, huella SHA-256 y lectura del catálogo en Latin-1 |
| `tests/test_features.py` | Unitarias | Preprocesamiento: conteo de salidas y llegadas, calendario, rezagos sin fuga de información, imputación, escalado y one-hot |
| `tests/test_metrics.py` | Unitarias | Cálculo de métricas: MAE, RMSE, WAPE y sesgo, por segmento y de las líneas base |
| `tests/test_inference.py` | Unitarias | Inferencia: `DemandForecaster.predict` con un modelo simulado (recorte de negativos, columnas, rechazo de un panel inválido) |
| `tests/test_modeling.py` | Unitarias y una de integración | Partición y validación cruzada temporal, pipelines de cada modelo, y entrenamiento con registro en MLflow |
| `tests/test_validation.py` | Unitarias | Validación del panel y de los CSV crudos |
| `tests/test_api.py` | Unitarias | API con `TestClient` y un modelo chico exportado en un directorio temporal: `/health`, `/predict` válido, estación desconocida (404), entrada inválida (422), estación-hora repetida (422) y modelo sin cargar (503) |
| `tests/test_integration.py` | Integración | Flujo completo: CSV crudos → limpieza → variables → validación → entrenamiento y Model Registry → predicción → métricas |

Las pruebas de integración llevan el marcador `integration` y son las que entrenan modelos.
Para correr solo una parte:

```bash
pytest -q -m "not integration"   # 46 pruebas unitarias, en segundos
pytest -q -m integration         # 6 pruebas de integración
```

## Servicio del modelo (Fase 3)

### Artefacto servido

| | |
|---|---|
| Modelo registrado | `models:/mibici-demanda-salidas/1` (alias `champion`) |
| Algoritmo | Gradient boosting por histogramas (`HistGradientBoostingRegressor`) |
| Métricas de prueba | MAE 0.862, RMSE 1.502 (`cv_rmse` 1.270) |
| Carpeta en el repositorio | `models/mibici-demanda-salidas/` |

`make export-model` (`mibici/modeling/export.py`) toma del Model Registry la versión con el
alias `champion` y deja en `models/mibici-demanda-salidas/`:

- `model/`: el pipeline en formato MLflow (`MLmodel` y `model.skops`).
- `estaciones.csv`: los atributos fijos de las 469 estaciones (`lat`, `lon`, `capacity`,
  `zone`, `pc1`, `pc2`, `pc3`, `station_cluster`).
- `model_info.json`: nombre, versión, alias, `model_uri`, `run_id`, hiperparámetros, métricas
  de validación y de prueba, versión (`v2.1`) y huella SHA-256 del panel, fecha de exportación
  y versiones de Python, scikit-learn y MLflow.

Esta carpeta sí se versiona en Git (pesa 2 MB): es lo que entra a la imagen Docker, así que
el servicio no necesita `mlflow.db`. Si se promueve otra versión a `champion`, se vuelve a
correr `make export-model`.

### API con FastAPI

El código está en `mibici/serving/`: `schemas.py` (Pydantic), `service.py`
(`ForecastService`) y `app.py` (FastAPI). Cada instancia trae la estación, la hora y las
salidas recientes; el servicio calcula el calendario con `DemandFeatureBuilder`, agrega los
atributos de `estaciones.csv`, valida el panel con `PanelValidator` y recorta los negativos.

```bash
make api        # uvicorn mibici.serving.app:app --port 8000
make openapi    # escribe docs/openapi.json
```

| Endpoint | Qué hace |
|---|---|
| `GET /health` | Estado y versión del modelo cargado |
| `GET /model` | El contenido de `model_info.json` |
| `POST /predict` | Pronóstico de 1 a 1,000 combinaciones estación-hora |
| `GET /docs` | Swagger UI con un ejemplo listo para probar |

Ejemplo:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "instances": [{"station_id": 51, "hour_ts": "2026-09-24T08:00:00",
                 "dep_lag_1": 32, "dep_lag_24": 31, "dep_lag_168": 25,
                 "dep_roll_mean_24": 15.0417}]}'
```

```json
{"model": {"name": "mibici-demanda-salidas", "version": 1, "model_uri": "models:/mibici-demanda-salidas/1"},
 "predictions": [{"station_id": 51, "hour_ts": "2026-09-24T08:00:00", "pronostico": 30.720685542405807}]}
```

Errores: 422 si la entrada no cumple el esquema (Pydantic) o si el panel no pasa
`PanelValidator` (por ejemplo, una estación-hora repetida), 404 si la estación no existe en
`estaciones.csv` y 503 si el modelo no está cargado. El detalle está en
[docs/docs/api.md](docs/docs/api.md).

### Imagen Docker

```bash
docker build -t ml-service:latest .
docker run -p 8000:8000 ml-service:latest
```

`make docker-build` y `make docker-run` corren esos mismos comandos. La imagen parte de
`python:3.11-slim`, corre con un usuario sin privilegios, instala solo
`requirements-serving.txt` (con `mlflow-skinny` en lugar de `mlflow`) y tiene un
`HEALTHCHECK` contra `/health`. Ocupa 862 MB en disco (190 MB comprimida).

### Imagen en Docker Hub

La imagen está publicada en
[luisadr18/mibici-demanda](https://hub.docker.com/r/luisadr18/mibici-demanda):

| Tag | Significado |
|---|---|
| `1.0.0` | Versión semántica de la imagen; coincide con la versión de la API (`API_VERSION`). Sirve `models:/mibici-demanda-salidas/1`. |
| `latest` | Apunta a la última versión publicada (hoy, `1.0.0`) |

Para correrla sin clonar el repositorio:

```bash
docker run -p 8000:8000 luisadr18/mibici-demanda:1.0.0
```

Convención: se sube un tag `MAYOR.MENOR.PARCHE` nuevo cada vez que cambia la imagen (el
parche para correcciones, el menor para un modelo nuevo o un endpoint nuevo que no rompe la
entrada, el mayor si cambia el esquema de entrada) y se mueve `latest` a esa versión:

```bash
docker tag ml-service:latest luisadr18/mibici-demanda:<version>
docker tag ml-service:latest luisadr18/mibici-demanda:latest
docker push luisadr18/mibici-demanda:<version>
docker push luisadr18/mibici-demanda:latest
```

## Organización del proyecto

```
├── Dockerfile           <- Imagen del servicio (docker build -t ml-service:latest .)
├── Makefile             <- Comandos: requirements, data, train, predict, test, lint, mlflow-ui, api, docker-build
├── README.md
├── data
│   ├── raw              <- CSV originales de MiBici (no versionados en git)
│   ├── interim          <- viajes_limpios_v1.parquet
│   ├── processed        <- panel_estacion_hora_v2.parquet, pronostico_salidas.parquet
│   └── external         <- Fuentes externas (por ahora vacía)
├── docs                 <- Documentación en mkdocs y openapi.json
├── models
│   └── mibici-demanda-salidas <- Modelo champion exportado: model/, estaciones.csv, model_info.json
├── notebooks            <- Notebooks de exploración que usan el paquete
├── references           <- Diccionario de datos y bitácora de cambios
├── reports
│   ├── figures          <- Figuras del EDA, de los modelos y capturas de MLflow
│   └── tables           <- Reporte de limpieza, líneas base y resultados de los modelos
├── requirements-serving.txt <- Dependencias de la imagen Docker
├── tests                <- Pruebas unitarias, de integración y de la API (pytest)
└── mibici               <- Código fuente del proyecto
    ├── config.py        <- Rutas, semilla, fecha de corte, parámetros y espacios de búsqueda
    ├── dataset.py       <- MiBiciLoader: descarga desde mibici.net, lectura y homologación de columnas
    ├── cleaning.py      <- TripCleaner: reglas de limpieza con reporte por regla
    ├── features.py      <- StationProfiler, DemandFeatureBuilder y build_preprocessor
    ├── plots.py         <- EDAReport y ModelReport: estadísticas y figuras
    ├── versioning.py    <- DatasetVersioner: registro de versiones de datos en MLflow
    ├── pipeline.py      <- Orquestación de la Fase 1 (datos)
    └── modeling
        ├── evaluation.py <- Partición temporal, validación cruzada y métricas
        ├── train.py      <- DemandExperiment: ajuste, evaluación, MLflow y Model Registry
        ├── predict.py    <- DemandForecaster: pronóstico con el modelo registrado
        └── export.py     <- ModelExporter: exporta el champion para servirlo
    └── serving
        ├── schemas.py    <- Esquemas de entrada y salida (Pydantic)
        ├── service.py    <- ForecastService: del cuerpo de la API al pronóstico
        ├── app.py        <- Aplicación FastAPI: /health, /model, /predict
        └── openapi.py    <- Escribe docs/openapi.json
```

Estructura basada en la plantilla Cookiecutter Data Science v2.

## Flujo de trabajo del equipo

- `master` es la rama estable y `Develop` la de integración.
- Cada tarea se trabaja en una rama propia (`feature/...`) y entra a `Develop` por pull
  request revisado por otro integrante. `Develop` pasa a `master` al cierre de cada fase.
- Antes de abrir el pull request: `make lint` y `make test`.
