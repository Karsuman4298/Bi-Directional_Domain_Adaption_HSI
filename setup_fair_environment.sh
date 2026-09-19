#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# A clean environment avoids mixing NumPy 1.x/2.x binary extensions from old runs.
VENV="${VENV:-.venv-fair}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
if [[ -e "$VENV" ]]; then
  echo "Environment already exists: $VENV. Activate it, or set VENV to a fresh directory."
  exit 2
fi
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install 'torch==2.8.0' --index-url "$TORCH_INDEX"
"$VENV/bin/python" -m pip install -r requirements-fair.txt
"$VENV/bin/python" -m pip check
"$VENV/bin/python" -m experiments.preflight
printf 'Activate with: source %s/bin/activate\n' "$VENV"
