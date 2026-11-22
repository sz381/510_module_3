ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE} AS project

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_ENABLE_TELEMETRY=false \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 app \
    && chown app:app /app

COPY config.yaml ./
COPY src/ ./src/
COPY api/ ./api/
COPY script/ ./script/
COPY data/processed/*.parquet ./data/processed/
COPY docker-entrypoint.sh /usr/local/bin/seattle
RUN chmod 0755 /usr/local/bin/seattle

USER app
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/seattle"]
CMD ["serve"]

FROM project AS trained
RUN /usr/local/bin/seattle train

FROM project AS prediction-api
COPY --from=trained --chown=app:app /app/models/ ./models/
