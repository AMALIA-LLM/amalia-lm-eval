#!/usr/bin/env bash

# Helper script to locate and source the project's .env file (at repo root)
# and provide exported variables for other run scripts.

# Determine the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Search upwards for .env starting from the script dir
SEARCH_DIR="$SCRIPT_DIR"
ROOT_ENV=""
while [ "$SEARCH_DIR" != "/" ]; do
  if [ -f "$SEARCH_DIR/.env" ]; then
    ROOT_ENV="$SEARCH_DIR/.env"
    break
  fi
  SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done

if [ -n "$ROOT_ENV" ]; then
  echo "🔎 Loading environment variables from $ROOT_ENV"
  # Export all variables from .env while sourcing it
  set -o allexport
  # shellcheck disable=SC1090
  source "$ROOT_ENV"
  set +o allexport
else
  echo "⚠️  .env not found in parent directories. Continuing without it."
fi

# Do NOT supply silent defaults for critical variables; require the env to provide them
# The run scripts themselves will decide whether DEFAULT_EVAL_OUTPUT_DIR is required
# but the Python environment activation is required here.

# Ensure ACTIVATE_ENV is set in .env and points to a real file
if [ -z "${ACTIVATE_ENV:-}" ]; then
  echo "❌ ERROR: ACTIVATE_ENV is not set in .env. Please set ACTIVATE_ENV to the path of your virtualenv activate script." >&2
  exit 1
fi

# Source the python environment activation script
echo "🔁 Activating Python environment via $ACTIVATE_ENV"
# shellcheck disable=SC1090
if [ -n "$ACTIVATE_ENV" ]; then
    if [ -f "$ACTIVATE_ENV" ]; then
        # It's a file (Standard venv/virtualenv)
        source "$ACTIVATE_ENV"
    elif [ -d "$ACTIVATE_ENV" ]; then
        # It's a directory (Conda env)
        # 1. Find where conda is installed
        CONDA_BASE_PATH=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE_PATH" ]; then
            source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
            conda activate "$ACTIVATE_ENV"
        else
            # Fallback: Just put the bin folder on the PATH
            export PATH="$ACTIVATE_ENV/bin:$PATH"
        fi
    else
        echo "⚠️ Warning: ACTIVATE_ENV path not found: $ACTIVATE_ENV"
    fi
fi

# Export variables for calling scripts; do not invent defaults
export DEFAULT_EVAL_OUTPUT_DIR
export HF_CACHE_REMOTE
export ACTIVATE_ENV

# print the variables for debugging
echo "✅ Environment setup complete with the following variables:"
echo "  DEFAULT_EVAL_OUTPUT_DIR=${DEFAULT_EVAL_OUTPUT_DIR:-<not set>}"
echo "  HF_CACHE_REMOTE=${HF_CACHE_REMOTE:-<not set>}"
echo "  ACTIVATE_ENV=${ACTIVATE_ENV:-<not set>}"
