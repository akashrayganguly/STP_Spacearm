#!/bin/bash
# =====================================================================
# Claude Code cloud environment for the spacearm project.
# In claude.ai/code open the environment menu (cloud icon) -> edit your environment:
#
#   Network access:  Custom, tick "Also include default list of common package managers",
#                    and add one line:   download.pytorch.org
#                    (optional: lets torch install as a ~200 MB CPU build instead of ~3 GB)
#   Environment variables (.env format):
#                    BASH_DEFAULT_TIMEOUT_MS=300000
#                    BASH_MAX_TIMEOUT_MS=1800000
#   Setup script:    paste this whole file.
#
# It runs on Ubuntu 24.04 (x86_64) before Claude starts. If it finishes in
# ~5 min the result is cached, so later sessions start with everything installed.
# It never exits non-zero (a failing setup script blocks the session);
# problems are printed as WARN lines and Claude fixes them in-session.
# =====================================================================
VENV="$HOME/.venvs/spacearm"
PY="$VENV/bin/python"

# 1) Python 3.11: PyBullet publishes Linux wheels only up to Python 3.11
#    (system Python on Ubuntu 24.04 is 3.12 and would compile PyBullet from source).
if [ ! -x "$PY" ]; then
  uv python install 3.11 && uv venv "$VENV" --python 3.11 --seed || echo "WARN: could not create the Python 3.11 venv"
fi

if [ -x "$PY" ]; then
  # 2) CPU-only PyTorch (~200 MB) when download.pytorch.org is allowed (Custom network access);
  #    otherwise the default PyPI build (larger download, also fine on CPU).
  uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu \
    || uv pip install --python "$PY" torch \
    || echo "WARN: torch install failed"

  # 3) Everything else (same set as environment.yml, from PyPI wheels).
  uv pip install --python "$PY" pybullet numpy scipy pyyaml matplotlib pandas tqdm pytest \
      "gymnasium>=1.1,<2" onnx onnxruntime imageio imageio-ffmpeg \
    || echo "WARN: package install failed"

  # 4) Activate the venv in every shell Claude opens.
  grep -q "venvs/spacearm/bin/activate" "$HOME/.bashrc" 2>/dev/null \
    || echo 'source "$HOME/.venvs/spacearm/bin/activate"' >> "$HOME/.bashrc"

  "$PY" -c "import pybullet, torch, gymnasium, onnxruntime; print('spacearm env OK: torch', torch.__version__)" \
    || echo "WARN: import check failed"
fi
exit 0
