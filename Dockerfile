# Imagen del servicio de pronostico MiBici (Fase 3).
#   docker build -t ml-service:latest .
#   docker run -p 8000:8000 ml-service:latest
# La API queda en http://localhost:8000 y su documentacion en /docs.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MODEL_DIR=/app/models/mibici-demanda-salidas

WORKDIR /app

# Usuario sin privilegios para correr el servicio
RUN useradd --create-home --uid 1000 app

# Dependencias primero: esta capa se reutiliza mientras no cambie el archivo
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

# Paquete del proyecto, sin dependencias (ya se instalaron arriba)
COPY pyproject.toml README.md LICENSE ./
COPY mibici/ mibici/
RUN pip install --no-deps .

# Modelo exportado con make export-model
COPY models/mibici-demanda-salidas/ models/mibici-demanda-salidas/

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "mibici.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
