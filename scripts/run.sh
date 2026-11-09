#!/usr/bin/env bash
# Hot-swappable drafter/loss runner.
#
#   scripts/run.sh --list
#   scripts/run.sh --check                       # one step per (method, loss)
#   scripts/run.sh --method dflash --loss dpace --steps 50
#   scripts/run.sh --method dflash --loss ce --device cpu --steps 5
#
# Point PY at the env that has torch + cupy:
#   PY=/home/direct11/anaconda3/envs/fi-bench/bin/python scripts/run.sh --check
set -euo pipefail

PY="${PY:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "error: '$PY' has no torch; set PY=/path/to/env/bin/python" >&2
  exit 1
fi

exec "$PY" "$ROOT/pipeline.py" "$@"
