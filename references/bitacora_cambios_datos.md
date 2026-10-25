# Bitácora de cambios a los datos

| Versión | Origen | Cambio | Filas |
|---|---|---|---|
| v0 | CSV jul, ago, sep 2026 | Concatenación y renombrado de columnas | 1,271,837 |
| v1 | v0 | Se eliminan 758 duplicados exactos (todos de septiembre) | 1,271,079 |
| v1 | v0 | Se eliminan 52,779 viajes de menos de 1 minuto | 1,218,300 |
| v1 | v0 | Se eliminan 8,871 viajes de menos de 2 minutos con origen igual a destino | 1,209,429 |
| v1 | v0 | Se eliminan 151 viajes de más de 180 minutos | 1,209,278 |
| v1 | v0 | Se eliminan 1,937 viajes con estación fuera del catálogo (estación 407) | 1,207,341 |
| v1 | v0 | 89 años de nacimiento fuera de 1940 a 2010 se convierten en nulo; género nulo pasa a NE | 1,207,341 |
| v1b | v0 | Igual que v1 pero con duración máxima de 48.6 min (regla de Tukey en escala log) | 1,204,974 |
| v2 | v1 | Agregación a panel estación-hora, features de calendario, rezagos, PCA y KMeans; se quitan las primeras 168 h por falta de rezagos | 956,760 |
