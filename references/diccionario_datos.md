# Diccionario de datos

## Viajes (data/raw/datos_abiertos_AAAA_MM.csv)

| Columna original | Columna en el proyecto | Tipo | Descripción |
|---|---|---|---|
| Viaje_Id | trip_id | entero | Identificador del viaje |
| Usuario_Id | user_id | entero | Identificador anónimo del usuario |
| Genero | gender | categoría | M, F; nulos o inválidos se codifican como NE |
| Año_de_nacimiento | birth_year | real | Año de nacimiento; fuera de 1940 a 2010 se vuelve nulo |
| Inicio_del_viaje | start_time | fecha-hora | Inicio del préstamo |
| Fin_del_viaje | end_time | fecha-hora | Anclaje de la bicicleta |
| Origen_Id | origin_id | entero | Estación de salida |
| Destino_Id | destination_id | entero | Estación de llegada |

Columnas derivadas en v1: source_month, duration_min, age, age_group, start_hour, is_round_trip.

## Estaciones (data/raw/nomenclatura_2026_09.csv)

| Columna original | Columna en el proyecto | Descripción |
|---|---|---|
| id | station_id | Identificador de estación |
| name | station_name | Clave y cruce de calles, por ejemplo (GDL-049) |
| latitude, longitude | lat, lon | Coordenadas |
| dpcapacity | capacity | Número de anclajes |

Columnas derivadas: zone (GDL, ZPN, TLQ) e is_retired (nombre inicia con "ex").

## Panel estación-hora (data/processed/panel_estacion_hora_v2.parquet)

| Columna | Descripción |
|---|---|
| departures | Variable objetivo: viajes que inician en la estación en esa hora |
| arrivals, net_flow | Llegadas y llegadas menos salidas |
| hour, dayofweek, is_weekend, is_holiday | Calendario |
| hour_sin, hour_cos, dow_sin, dow_cos | Codificación cíclica de hora y día |
| dep_lag_1, dep_lag_24, dep_lag_168 | Salidas 1 h, 24 h y 168 h antes |
| dep_roll_mean_24 | Promedio de salidas de las 24 horas previas |
| lat, lon, capacity, zone | Atributos de la estación |
| pc1, pc2, pc3, station_cluster | Componentes PCA y grupo KMeans del perfil horario |
