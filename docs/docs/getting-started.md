# Cómo reproducir el proyecto

## 1. Entorno

Se requiere Python 3.11 o superior.

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # en Windows: .venv\Scripts\activate
make requirements
```

## 2. Datos crudos

Los CSV no se suben al repositorio por su tamaño. Se necesitan estos cuatro archivos en `data/raw/`:

- `datos_abiertos_2026_07.csv`
- `datos_abiertos_2026_08.csv`
- `datos_abiertos_2026_09.csv`
- `nomenclatura_2026_10.csv`

La forma más sencilla es descargarlos con el cargador del paquete, que localiza los enlaces en
<https://www.mibici.net/es/datos-abiertos/> y guarda los archivos en `data/raw/`. Si un archivo ya
existe, se omite la descarga:

```bash
python -c "from mibici.dataset import MiBiciLoader as L; l = L(); l.download_trips(2026, [7, 8, 9]); l.download_stations()"
```

También puedes descargarlos a mano desde el portal y colocarlos en `data/raw/`.

## 3. Fase 1: datos

```bash
make data
```

Genera `data/interim/viajes_limpios_v1.parquet` y `data/processed/panel_estacion_hora_v2.parquet`,
las figuras del análisis exploratorio y un run de MLflow por cada versión del dataset
(experimento `mibici-fase1-datos`).

## 4. Fase 2: modelos

```bash
make train
make predict
```

`make train` ajusta los modelos candidatos con validación temporal, registra cada configuración
en el experimento `mibici-fase2-modelos` y deja los mejores en el Model Registry con el nombre
`mibici-demanda-salidas`. `make predict` carga la versión con alias `champion` y escribe el
pronóstico en `data/processed/pronostico_salidas.parquet`.

## 5. Revisar los resultados

```bash
make mlflow-ui
```

Abre <http://localhost:5000>. En *Experiments* están los runs para comparar y en *Models* las
versiones registradas.

## 6. Pruebas

```bash
make test
make lint
```
