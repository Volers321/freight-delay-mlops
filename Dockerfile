FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd --create-home --uid 10001 app
USER app

ENV MLFLOW_TRACKING_URI=http://mlflow:5000 \
    FDM_PREDICTION_LOG=/data/predictions.jsonl
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(0 if b'\"ok\"' in urllib.request.urlopen('http://localhost:8000/health').read() else 1)"
CMD ["uvicorn", "fdm.api:app", "--host", "0.0.0.0", "--port", "8000"]
