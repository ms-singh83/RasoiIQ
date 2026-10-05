#!/usr/bin/env bash
# One command: set up (first time only) and start RasoiIQ at http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.11 python3.12; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "RasoiIQ needs Python 3.11 or 3.12 (torch 2.2.2, the last Intel-Mac PyTorch, has no 3.13 build)."
  echo "On a Mac:  brew install python@3.11   then run ./run.sh again."
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "First run: creating .venv with $PY and installing packages (about 1 GB, a few minutes)..."
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip >/dev/null
  .venv/bin/pip install -r requirements.txt
fi

if [ ! -f orders.csv ] && [ ! -f data/sample_orders.csv ]; then
  .venv/bin/python scripts/generate_sample.py
fi
if [ -f orders.csv ]; then echo "Using your orders.csv"; else echo "No orders.csv found, using SAMPLE data (data/sample_orders.csv)"; fi

export TABPFN_DISABLE_TELEMETRY=1
PORT="${PORT:-8000}"
echo "Open http://localhost:$PORT  (first forecast downloads the 44 MB TabPFN model once, no login)"
( sleep 3; command -v open >/dev/null 2>&1 && open "http://localhost:$PORT" ) >/dev/null 2>&1 &
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
