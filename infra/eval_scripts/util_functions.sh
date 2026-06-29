#!/usr/bin/env bash

# Shared functions used across all run_*.sh evaluation scripts

# --- Shared Constants ---
export MAX_SLURM_JOBS=366
export JOBS_PER_CHECKPOINT=22
export BUFFER_JOBS=2  # Safety buffer

# --- Function to check if path is already a checkpoint ---
is_checkpoint_path() {
    local path="$1"

    # Check if path contains "checkpoint-" in its name
    if [[ "$path" =~ checkpoint-[0-9]+ ]]; then
        return 0
    fi

    # Check if path has .safetensors files and no checkpoint-* subdirectories
    if [ -n "$(find "$path" -maxdepth 1 -name "*.safetensors" 2>/dev/null)" ] && \
       [ -z "$(find "$path" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null)" ]; then
        return 0
    fi

    return 1
}

# --- Function to get checkpoint directories sorted by number (newest first) ---
get_sorted_checkpoints() {
    local model_path="$1"

    # Find all checkpoint-* directories and sort numerically (descending)
    find "$model_path" -maxdepth 1 -type d -name "checkpoint-*" | \
        sed 's/.*checkpoint-//' | \
        sort -rn | \
        sed "s|^|$model_path/checkpoint-|"
}

# --- Function to check if we can submit more jobs ---
can_submit_jobs() {
    local current_jobs=$(squeue | wc -l)
    # Subtract 1 for the header line
    current_jobs=$((current_jobs - 1))

    local total_after_submit=$((current_jobs + JOBS_PER_CHECKPOINT + BUFFER_JOBS))

    if [ $total_after_submit -ge $MAX_SLURM_JOBS ]; then
        return 1  # Cannot submit
    else
        return 0  # Can submit
    fi
}

# --- Function to wait for jobs to complete ---
wait_for_job_slots() {
    echo "⏳ Queue full. Waiting for jobs to complete..."
    while ! can_submit_jobs; do
        sleep 60  # Check every minute
    done
    echo "✅ Queue has space. Continuing..."
}

# --- Function to check if a path is a valid LoRA model ---
is_lora_model() {
    local path="$1"

    # Check if adapter_model.safetensors exists in the path
    if [ -f "$path/adapter_model.safetensors" ]; then
        return 0
    fi

    return 1
}

# --- Function to check if output directory should be skipped ---
should_skip_output_dir() {
    local outdir="$1"
    local skip_existing="${2:-true}"  # Default to true for backward compatibility

    # If skip_existing is false, never skip
    if [ "$skip_existing" = "false" ]; then
        return 1  # Do not skip
    fi

    # Check if output directory exists and has result files
    if [ -d "$outdir" ] && [ -n "$(find "$outdir" -name '*.json' -o -name '*.jsonl' 2>/dev/null)" ]; then
        return 0  # Should skip
    fi

    return 1  # Do not skip
}