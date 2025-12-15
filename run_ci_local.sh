#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_ci_local.sh – simulate the GitHub Actions CI pipeline locally.
# ---------------------------------------------------------------------------
# The script performs the same essential checks defined in
# .github/workflows/ci.yml; it is network-free if NumPy is already available,
# but will use the network to install NumPy into `.venv` if needed.
# It is handy when you want to verify the project passes BEFORE pushing.
#
# What it does:
# 1. Ensures NumPy is available (installs it into a venv under .venv if not).
# 2. Builds the Lean project (using Lake).
# 3. Executes the Lean × NumPy cross-check.
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)

# ---------------------------------------------------------------------------
# Python environment – create a lightweight venv under .venv if needed.
# ---------------------------------------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ Python 3 is required but was not found in PATH." >&2
  exit 1
fi

# Attempt to use system Python with its packages first.
if python3 - <<'PY'
import importlib
import sys

try:
    importlib.import_module("numpy")
    sys.exit(0)
except ImportError:
    sys.exit(1)
PY
then
  :
else
  # NumPy not available globally – fall back to venv and try to install.
  VENV_DIR="$PROJECT_ROOT/.venv"

  # If an existing `.venv` is present but broken (e.g. points to a missing
  # interpreter), recreate it rather than failing later with "no such file".
  if [ -d "$VENV_DIR" ] && [ ! -x "$VENV_DIR/bin/python3" ]; then
    echo "⚠️  Existing .venv appears to be broken; recreating it."
    rm -rf "$VENV_DIR"
  fi

  if [ ! -d "$VENV_DIR" ]; then
    echo "🔹 Creating Python virtual environment (.venv)"
    python3 -m venv "$VENV_DIR"
  fi

  source "$VENV_DIR/bin/activate"

  python3 -m pip --quiet install --upgrade pip >/dev/null
  echo "🔹 Installing NumPy into .venv … (requires internet)"
  python3 -m pip install numpy || {
    echo "❌ Failed to install NumPy. Ensure internet access or pre-install it system-wide." >&2
    exit 1
  }
fi

# ---------------------------------------------------------------------------
# Build Lean project (will use cached .lake artefacts if present).
# ---------------------------------------------------------------------------

echo "🔹 Building Lean project with Lake …"
lake build >/dev/null

# ---------------------------------------------------------------------------
# Run cross-check.
# ---------------------------------------------------------------------------

echo "🔹 Running Lean × NumPy cross-check …"
python3 cross_check_numpy.py

echo "✅ Local CI succeeded!"
