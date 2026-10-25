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

## Fase 1: datos

La Fase 1 cubre la preparación de datos: carga, limpieza, análisis exploratorio,
construcción de variables y versionado de cada versión del dataset en MLflow.

Datos usados: viajes de julio, agosto y septiembre de 2026 y el catálogo de estaciones
`nomenclatura_2026_09.csv`, descargados de https://mibici.net/es/datos-abiertos/.
Los CSV se colocan en `data/raw/` (no se suben al repositorio por su tamaño).

## Cómo reproducir

```bash
make create_environment
make requirements
make data      # corre mibici/pipeline.py: limpieza, features, EDA y versiones en MLflow
make test
make mlflow-ui # abre la interfaz de MLflow en http://localhost:5000
```

## Versiones del dataset (registradas en MLflow)

| Versión | Etapa | Descripción |
|---|---|---|
| v0 | raw | Viajes crudos concatenados de los tres meses |
| v1 | interim | Viajes limpios, duración máxima de 180 min (criterio de dominio) |
| v1b | interim | Variante experimental, duración máxima por regla de Tukey sobre log(duración) |
| v2 | processed | Panel estación-hora con variables de calendario, rezagos y perfil de estación |

Cada run guarda parámetros de limpieza, métricas de filas eliminadas, esquema, huella
SHA-256 del contenido y el archivo parquet resultante.

## Organización del proyecto

```
├── Makefile             <- Comandos: requirements, data, test, lint, mlflow-ui
├── README.md
├── data
│   ├── raw              <- CSV originales de MiBici (no versionados en git)
│   ├── interim          <- viajes_limpios_v1.parquet
│   └── processed        <- panel_estacion_hora_v2.parquet
├── notebooks            <- Notebook de exploración que usa el paquete refactorizado
├── references           <- Diccionario de datos y bitácora de cambios
├── reports
│   ├── figures          <- Figuras del EDA y capturas de MLflow
│   └── tables           <- Reporte de limpieza, resumen del EDA, líneas base
├── tests                <- Pruebas unitarias de limpieza y versionado
└── mibici               <- Código fuente del proyecto
    ├── config.py        <- Rutas y parámetros (CleaningConfig, FeatureConfig)
    ├── dataset.py       <- MiBiciLoader: lectura y homologación de columnas
    ├── cleaning.py      <- TripCleaner: reglas de limpieza con reporte por regla
    ├── features.py      <- StationProfiler (PCA + KMeans) y DemandFeatureBuilder
    ├── plots.py         <- EDAReport: estadísticas y figuras
    ├── versioning.py    <- DatasetVersioner: registro de versiones en MLflow
    └── pipeline.py      <- Orquestación de la Fase 1
```

Estructura basada en la plantilla Cookiecutter Data Science v2.
