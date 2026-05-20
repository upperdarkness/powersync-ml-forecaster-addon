#!/usr/bin/env bash
set -euo pipefail
mkdir -p /share/powersync_ml_forecaster
cd /app
exec python3 standalone_runner.py
