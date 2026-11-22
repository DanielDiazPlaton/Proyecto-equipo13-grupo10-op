"""Prueba de humo de la API de pronostico ya corriendo en un contenedor.

Llama a los endpoints por HTTP, revisa los codigos de error documentados y compara el
pronostico de la imagen construida en esta corrida con el de la imagen publicada en Docker
Hub. Solo usa la biblioteca estandar, para correr con el Python del runner sin instalar nada.

Uso:
    python .github/scripts/smoke_api.py --built http://127.0.0.1:8000
    python .github/scripts/smoke_api.py --built http://127.0.0.1:8000 \
        --published http://127.0.0.1:8001
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# El ejemplo del README y de /docs: estacion 51, jueves 24 de septiembre de 2026.
EXAMPLE = {
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
# Pronostico de ese ejemplo con cada version del modelo, calculado fuera del contenedor
# (Windows, Python 3.12). Si se exporta otra version, se agrega aqui su referencia.
REFERENCE = {"models:/mibici-demanda-salidas/1": [30.720685542405807, 34.50356440913216]}
TOLERANCE = 1e-6


def call(base_url: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    """Devuelve el codigo HTTP y el JSON de la respuesta, tambien cuando es un error."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    request = urllib.request.Request(base_url + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


class Report:
    """Junta el resultado de cada revision para imprimirlo y decidir el codigo de salida."""

    def __init__(self):
        self.rows: list[tuple[str, str, bool, str]] = []

    def check(self, image: str, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((image, name, bool(ok), detail))
        print(f"[{'ok' if ok else 'FALLA'}] {image}: {name} {detail}".rstrip())

    @property
    def failed(self) -> bool:
        return any(not ok for _, _, ok, _ in self.rows)

    def markdown(self) -> str:
        lines = ["| Imagen | Revisión | Resultado | Detalle |", "|---|---|---|---|"]
        for image, name, ok, detail in self.rows:
            lines.append(f"| {image} | {name} | {'✅' if ok else '❌'} | {detail} |")
        return "\n".join(lines) + "\n"


def check_image(report: Report, image: str, base_url: str) -> list[float] | None:
    """Revisa una imagen y devuelve su pronostico del ejemplo (None si /predict fallo)."""
    status, health = call(base_url, "/health")
    report.check(image, "GET /health", status == 200 and health.get("status") == "ok", str(health))

    status, info = call(base_url, "/model")
    model_uri = info.get("model_uri", "")
    report.check(
        image, "GET /model", status == 200 and model_uri.startswith("models:/"), model_uri
    )

    status, body = call(base_url, "/predict", EXAMPLE)
    forecasts = [p["pronostico"] for p in body.get("predictions", [])]
    ok = status == 200 and len(forecasts) == len(EXAMPLE["instances"])
    report.check(image, "POST /predict", ok, str(forecasts))
    if not ok:
        return None
    report.check(image, "pronóstico no negativo", all(value >= 0 for value in forecasts))

    reference = REFERENCE.get(model_uri)
    if reference is None:
        print(f"::notice::{image}: no hay pronostico de referencia para {model_uri}")
    else:
        gap = max(abs(a - b) for a, b in zip(forecasts, reference))
        report.check(
            image,
            "igual al pronóstico fuera del contenedor",
            gap <= TOLERANCE,
            f"diferencia máxima {gap:.2e}",
        )

    instance = EXAMPLE["instances"][0]
    errors = [
        ("404 con estación desconocida", 404, [{**instance, "station_id": 99999}]),
        ("422 con hora que no es exacta", 422, [{**instance, "hour_ts": "2026-09-24T08:30:00"}]),
        ("422 con rezago negativo", 422, [{**instance, "dep_lag_1": -1}]),
        ("422 con estación-hora repetida", 422, [instance, instance]),
        ("422 sin instancias", 422, []),
    ]
    for name, expected, instances in errors:
        status, _ = call(base_url, "/predict", {"instances": instances})
        report.check(image, name, status == expected, f"HTTP {status}")

    status, schema = call(base_url, "/openapi.json")
    report.check(
        image,
        "esquema OpenAPI con /predict",
        status == 200 and "/predict" in schema.get("paths", {}),
    )
    return forecasts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--built", required=True, help="URL de la imagen construida")
    parser.add_argument("--published", help="URL de la imagen publicada en Docker Hub")
    args = parser.parse_args()

    report = Report()
    built = check_image(report, "construida", args.built)
    if args.published:
        published = check_image(report, "publicada", args.published)
        if built is not None and published is not None:
            gap = max(abs(a - b) for a, b in zip(built, published))
            report.check(
                "ambas",
                "la imagen publicada da el mismo pronóstico que la construida",
                gap <= TOLERANCE,
                f"diferencia máxima {gap:.2e}",
            )

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write("## Prueba de humo de la API en contenedor\n\n" + report.markdown())
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
