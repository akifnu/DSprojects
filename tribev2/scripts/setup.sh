#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Creating virtual environment at ${VENV_DIR}"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing TRIBE v2 capability harness"
python -m pip install -r requirements.txt
python -m pip install -e .

echo "==> Running environment check"
python scripts/check_environment.py

echo "Setup complete."
echo "Colab (zero setup): tribev2/notebooks/Kahneman_Framing_RCT.ipynb"
