#!/bin/sh
set -eu

case "$1" in
    train)
        exec python -m src.train_serving
        ;;
    serve)
        shift
        exec python -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
