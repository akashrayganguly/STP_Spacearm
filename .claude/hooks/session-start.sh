#!/bin/bash
# SessionStart hook (cloud sessions only): make the Python 3.11 venv the default
# `python` for every command Claude runs, and install this repo in editable mode.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

VENV="$HOME/.venvs/spacearm"
PY="$VENV/bin/python"
cd "$CLAUDE_PROJECT_DIR"

# The environment's setup script normally builds the venv; build it here if it is missing.
if [ ! -x "$PY" ]; then
  bash cloud/setup.sh
fi

# Editable install of the repo (fast when already installed).
uv pip install --quiet --python "$PY" -e . || "$PY" -m pip install --quiet -e .

# Put the venv first on PATH for the rest of the session.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export VIRTUAL_ENV=\"$VENV\"" >> "$CLAUDE_ENV_FILE"
  echo "export PATH=\"$VENV/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi

"$PY" -c "import spacearm, pybullet, torch; print('spacearm venv ready: python', __import__('sys').version.split()[0], '| torch', torch.__version__)"
