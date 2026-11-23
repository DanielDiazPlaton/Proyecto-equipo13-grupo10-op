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

Se requiere Python 3.11.

```bash
make create_environment        # crea .venv
source .venv/bin/activate      # en Windows: .venv\Scripts\activate
make requirements
make data       # Fase 1: limpieza, features, EDA y versiones del dataset en MLflow
make train      # Fase 2: ajuste, evaluación y registro de modelos en MLflow
make predict    # pronóstico con el modelo champion del Model Registry
make test
make mlflow-ui  # interfaz de MLflow en http://localhost:5000
```

Datos usados: viajes de julio, agosto y septiembre de 2026 y el catálogo de estaciones
`nomenclatura_2026_10.csv`, descargados de https://mibici.net/es/datos-abiertos/.
Los cuatro CSV se colocan en `data/raw/` (no se suben al repositorio por su tamaño).

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
6. **Seguimiento en MLflow** (experimento `mibici-fase2-modelos`): un run por modelo con
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

## Organización del proyecto

```
├── Makefile             <- Comandos: requirements, data, train, predict, test, lint, mlflow-ui
├── README.md
├── data
│   ├── raw              <- CSV originales de MiBici (no versionados en git)
│   ├── interim          <- viajes_limpios_v1.parquet
│   ├── processed        <- panel_estacion_hora_v2.parquet, pronostico_salidas.parquet
│   └── external         <- Fuentes externas (por ahora vacía)
├── docs                 <- Documentación en mkdocs
├── models               <- Modelos exportados; los pipelines viven en el Model Registry de MLflow
├── notebooks            <- Notebooks de exploración que usan el paquete
├── references           <- Diccionario de datos y bitácora de cambios
├── reports
│   ├── figures          <- Figuras del EDA, de los modelos y capturas de MLflow
│   └── tables           <- Reporte de limpieza, líneas base y resultados de los modelos
├── tests                <- Pruebas unitarias de limpieza, versionado y modelado
└── mibici               <- Código fuente del proyecto
    ├── config.py        <- Rutas, semilla, fecha de corte, parámetros y espacios de búsqueda
    ├── dataset.py       <- MiBiciLoader: lectura y homologación de columnas
    ├── cleaning.py      <- TripCleaner: reglas de limpieza con reporte por regla
    ├── features.py      <- StationProfiler, DemandFeatureBuilder y build_preprocessor
    ├── plots.py         <- EDAReport y ModelReport: estadísticas y figuras
    ├── versioning.py    <- DatasetVersioner: registro de versiones de datos en MLflow
    ├── pipeline.py      <- Orquestación de la Fase 1 (datos)
    └── modeling
        ├── evaluation.py <- Partición temporal, validación cruzada y métricas
        ├── train.py      <- DemandExperiment: ajuste, evaluación, MLflow y Model Registry
        └── predict.py    <- DemandForecaster: pronóstico con el modelo registrado
```

Estructura basada en la plantilla Cookiecutter Data Science v2.

## Flujo de trabajo del equipo

- `master` es la rama estable y `Develop` la de integración.
- Cada tarea se trabaja en una rama propia (`feature/...`) y entra a `Develop` por pull
  request revisado por otro integrante. `Develop` pasa a `master` al cierre de cada fase.
- Antes de abrir el pull request: `make lint` y `make test`.
