"""Escribe el esquema OpenAPI de la API en docs/openapi.json (make openapi).

No necesita el modelo: el esquema sale de los endpoints y de los modelos de Pydantic.

Uso:
    python -m mibici.serving.openapi
"""

import json
from pathlib import Path

from loguru import logger
import typer

from mibici.config import OPENAPI_FILE
from mibici.serving.app import app as api

app = typer.Typer()


@app.command()
def main(output: Path = OPENAPI_FILE):
    output.parent.mkdir(parents=True, exist_ok=True)
    schema = json.dumps(api.openapi(), indent=2, ensure_ascii=False)
    output.write_text(schema + "\n", encoding="utf-8")
    logger.success(f"Esquema OpenAPI guardado en {output}")


if __name__ == "__main__":
    app()
