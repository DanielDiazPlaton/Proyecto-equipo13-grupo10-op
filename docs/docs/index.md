# mibici-mlops

Proyecto del curso Operaciones de aprendizaje automático (Gpo 10), Tecnológico de Monterrey,
Equipo 13.

## Descripción

MiBici es el sistema de bicicletas públicas del Área Metropolitana de Guadalajara. El proyecto
pronostica cuántos viajes inician en cada estación en cada hora, para que el equipo de operación
pueda redistribuir bicicletas antes de que una estación se quede vacía o sin espacios para anclar.

Los datos son los archivos mensuales de viajes y el catálogo de estaciones publicados en
<https://www.mibici.net/es/datos-abiertos/>.

## Fases

| Fase | Contenido | Punto de entrada |
|---|---|---|
| 1. Datos | Limpieza, análisis exploratorio, panel estación-hora y versiones del dataset en MLflow | `make data` |
| 2. Modelado | Pipeline de scikit-learn, ajuste de hiperparámetros, seguimiento en MLflow y Model Registry | `make train`, `make predict` |

## Comandos

El `Makefile` reúne los puntos de entrada del proyecto. `make help` muestra la lista completa.

| Comando | Qué hace |
|---|---|
| `make requirements` | Instala las dependencias |
| `make data` | Construye las versiones v0, v1, v1b y v2 del dataset |
| `make train` | Ajusta, evalúa y registra los modelos |
| `make predict` | Genera el pronóstico con el modelo `champion` del Model Registry |
| `make test` | Corre las pruebas unitarias |
| `make lint` | Revisa el estilo con ruff |
| `make mlflow-ui` | Abre la interfaz de MLflow |
