#!/bin/bash
#SBATCH --job-name=streamlit_viz
#SBATCH --output=streamlit_viz_%j.log
#SBATCH --error=streamlit_viz_%j.log
#SBATCH --time=0
# SBATCH --partition=general
#SBATCH --nodelist=nscluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2


# Determine script directory so .env can live next to this script
ENV_FILE="../.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading configuration from $ENV_FILE"
    # export all variables loaded from .env
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
else
    echo "ERROR: Required .env file not found at $ENV_FILE"
    exit 1
fi

# --- REQUIRED VARIABLES (no defaults) ---
REQUIRED_VARS=(REMOTE_USER REMOTE_HOST REMOTE_PATH LOCAL_DIR COPY_INTERVAL STREAMLIT_PORT)
missing=()
for v in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!v}" ]; then
        missing+=("$v")
    fi
done

if [ ${#missing[@]} -ne 0 ]; then
    echo "ERROR: Missing required variables in $ENV_FILE: ${missing[*]}"
    exit 1
fi

# Basic validation for numeric values
if ! [[ "$COPY_INTERVAL" =~ ^[0-9]+$ ]] || [ "$COPY_INTERVAL" -le 0 ]; then
    echo "ERROR: COPY_INTERVAL must be a positive integer (got: $COPY_INTERVAL)"
    exit 1
fi

if ! [[ "$STREAMLIT_PORT" =~ ^[0-9]+$ ]] || [ "$STREAMLIT_PORT" -le 0 ] || [ "$STREAMLIT_PORT" -gt 65535 ]; then
    echo "ERROR: STREAMLIT_PORT must be an integer between 1 and 65535 (got: $STREAMLIT_PORT)"
    exit 1
fi

# --- ENVIRONMENT SETUP ---
eval "$(conda shell.bash hook)"

conda init bash

conda activate py312_amalia

mkdir -p "$LOCAL_DIR"

echo "Starting periodic sync + Streamlit"
echo "Local dir: $LOCAL_DIR"

# --- START STREAMLIT ---
streamlit run --server.port "$STREAMLIT_PORT" --server.address 0.0.0.0 streamlit_viz.py -- --folder_path "$LOCAL_DIR" &
STREAMLIT_PID=$!

echo "Streamlit PID = $STREAMLIT_PID"
echo "Starting infinite periodic copy loop..."

# --- INFINITE LOOP ---
while true; do
    echo "[$(date)] Syncing files from remote..."

    # Prefer rsync (efficient, resumable). If rsync isn't available, fall back to scp -r.
    if command -v rsync >/dev/null 2>&1; then
        echo "Using rsync to sync ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/ -> ${LOCAL_DIR}/"
        rsync -avz --delete "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" "$LOCAL_DIR/"
        RC=$?
    else
        echo "rsync not found, falling back to scp -r"
        # This will copy the remote path (file or directory) into LOCAL_DIR.
        scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" "$LOCAL_DIR/"
        RC=$?
    fi

    if [[ $RC -eq 0 ]]; then
        echo "[$(date)] Sync OK"
    else
        echo "[$(date)] WARNING: Sync failed (rc=$RC)"
    fi

    sleep "$COPY_INTERVAL"
done

wait $STREAMLIT_PID
