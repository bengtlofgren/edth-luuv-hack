#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck source=/dev/null
. .venv/bin/activate
python -m pip install -q -r requirements.txt

if [ ! -d "frontend/web/node_modules" ]; then
  (cd frontend/web && npm ci)
fi

if [ ! -f "frontend/web/dist/index.html" ] || [ "frontend/web/src/ws.ts" -nt "frontend/web/dist/index.html" ]; then
  (cd frontend/web && npm run build)
fi

cat <<EOF
Command center starting on http://$HOST:$PORT
Planner: http://$HOST:$PORT/planner/
DVL adapter example: python scripts/feed_dvl.py --csv dvl.csv --speed 1
EOF

exec python -m uvicorn src.server:app --host "$HOST" --port "$PORT"
