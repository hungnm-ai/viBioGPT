#!/bin/bash
gunicorn "api.routers:app" --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8089 --reload --timeout 600 --log-level=INFO
