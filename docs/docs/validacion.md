# Validación de los datos

El proyecto realiza las validaciones en dos momentos antes de usarlos: primero los CSV crudos, justo después de descargarlo y en el panel estación‑hora, justo antes de entrenar o pronosticar. Las reglas están en `mibici/validation.py` y los valores esperados están en `mibici/config.py`.

## 1. Datos crudos: `make check-raw`

Ejecuta el comando justo después de descargar los archivos:

```bash
make check-raw
```

| Revisión | Si falla |
|---|---|
| Los tres CSV de viajes y el catálogo están en `data/raw/` | Error: indica cuáles faltan |
| Cada CSV de viajes tiene las columnas que espera el cargador | Error: MiBici cambió el formato |
| El catálogo tiene id, nombre, coordenadas y capacidad, sin ids repetidos | Error |
| El catálogo tiene 470 estaciones, como el del proyecto | Error (ver abajo) |
| Si tiene 470, la huella SHA‑256 del archivo coincide con la del catálogo del proyecto | Error (ver abajo) |
| Las estaciones llevan el municipio (GDL, ZPN o TLQ) en el nombre | Advertencia: en el panel aparecen como `OTRA` |
| Los tres meses suman 1 271 837 viajes | Error: probablemente MiBici volvió a publicar algún mes |
| La huella del contenido coincide con la de la versión v0 (`ae874bbb1ee3…`) | Error: el contenido cambió aunque el número de filas sea igual |

El comando también muestra el tamaño y la huella SHA‑256 de cada archivo. De esta forma es mucho mas sencillo notar si llega una versión nueva de los datos. Las huellas esperadas de todas las versiones (v0, v1, v1b y v2.1) están en `EXPECTED_FINGERPRINTS`. Esas mismas huellas guarda cada ejecución del experimento `mibici-fase1-datos` en la etiqueta `data.sha256`.

El resultado de cada regla queda en `reports/tables/reporte_validacion_crudos.csv`, con las columnas `regla`, `resultado` (ok, error o advertencia), `casos` y `nota`, igual que el reporte de limpieza.

### Por qué se revisa el catálogo

MiBici solo publica el catálogo de estaciones más reciente, y `download_stations()` siempre lo guarda como `nomenclatura_2026_10.csv`. Cuando MiBici publique uno nuevo, el archivo seguirá llamándose octubre pero tendrá otro contenido. Ya ocurrió una vez: el catálogo de septiembre tenía 484 estaciones y el de octubre 470.

Del catálogo extraemos la latitud, la longitud, la capacidad y el municipio de cada estación.
Las variables del modelo cambian cuando se usa un catálogo distinto. Por eso, al cambiar el catálogo también cambian los datos de entrenamiento. Si no se revisa, el cambio solo aparece en la huella del panel, que se muestra como advertencia por los decimales del PCA (sección 2) y pasa desapercibido.

La revisión tiene dos partes:

- **Número de estaciones (470).** Muestra un mensaje fácil de leer cuando el catálogo cambió.  
- **Huella SHA‑256 del archivo (`2da38dd60dcc…`).** Detecta cualquier otro cambio: capacidades, coordenadas o nombres distintos con el mismo número de estaciones.

Si la revisión falla, busca la copia de `nomenclatura_2026_10.csv` que usa el proyecto (otro integrante la tiene). Si el equipo decide usar el catálogo nuevo, cambia `STATIONS_FILE`, `EXPECTED_STATIONS` y `EXPECTED_STATIONS_FILE_SHA256` en `config.py`, y registra una versión nueva del panel en `references/bitacora_cambios_datos.md`.

## 2. Panel estación‑hora: `make validate`

El panel se valida en tres momentos:

- **`make data`:** el panel no se guarda si no pasa la validación. El reporte queda en `reports/tables/reporte_validacion_panel.csv` y se guarda como artefacto del run de la versión v2.1.  
- **`make train`:** `DemandExperiment` valida el panel antes de crear cualquier run en MLflow. Cada run del modelo guarda `reporte_validacion_panel.csv` y la etiqueta `data.validacion`, que también se lleva a las versiones del Model Registry.  
- **`make predict`:** `DemandForecaster` valida el panel que va a pronosticar, sin exigir la variable objetivo.

`make validate` corre la misma validación por separado y deja el reporte en `reports/tables/reporte_validacion_panel.csv`.

Las reglas están en `PanelValidator` y sus límites en `PanelValidationConfig`:

| Revisión | Tipo |
|---|---|
| Están todas las columnas que usa el modelo, la llave estación‑hora y la variable objetivo | Error |
| Cada columna tiene el tipo esperado (por ejemplo, `station_cluster` como texto) | Error |
| No hay filas repetidas de estación‑hora ni nulos en la llave | Error |
| `hour_ts` es una hora exacta y coincide con la columna `hour` | Error |
| La variable objetivo no tiene nulos ni valores negativos | Error |
| Los rezagos y el promedio móvil no son negativos | Error |
| `is_weekend` e `is_holiday` valen 0 o 1; senos y cosenos están en [-1, 1] | Error |
| Ninguna variable del modelo tiene más de 5 % de nulos | Error |
| Hay nulos por debajo del 5 % (el pipeline los imputa con la mediana) | Advertencia |
| Coordenadas fuera del área metropolitana de Guadalajara | Advertencia |
| Estaciones con capacidad cero o negativa, o municipio no reconocido | Advertencia |
| La huella del panel no coincide con la registrada para la v2.1 | Advertencia |

La huella del panel se muestra como advertencia y no como error porque la huella depende de las componentes PCA. Las componentes PCA cambian en los últimos decimales entre equipos. Con los datos reales la huella salió `76f3621b3722` en lugar de `6c573764ce04`. Las dimensiones y las métricas de Ridge son iguales hasta el decimal 12.  

**Por qué importa:** varios de estos errores no detienen el pipeline o lo detienen con un mensaje difícil de entender. Por ejemplo, el modelo registrado se entrenó con `station_cluster` como texto (`"0"`, `"1"`, `"2"`). Si un panel nuevo lo trae como número, el `OneHotEncoder` falla a mitad del pronóstico con `TypeError: ufunc 'isnan' not supported`. Una variable objetivo negativa o filas repetidas ni siquiera producen un error. La validación detecta estos problemas antes de entrenar o pronosticar y dice qué regla falló.