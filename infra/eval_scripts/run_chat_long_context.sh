#!/bin/bash

umask 002

# Source environment helper to load .env and activate Python env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/util_env.sh"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/util_functions.sh"

# --- Output Directory Configuration ---
# Do not silently default; require the .env to set DEFAULT_EVAL_OUTPUT_DIR or the user to pass -o
if [ -z "${DEFAULT_EVAL_OUTPUT_DIR:-}" ]; then
    echo "❌ ERROR: DEFAULT_EVAL_OUTPUT_DIR is not set in .env. Please set it or pass -o/--output on the command line." >&2
    exit 1
fi

OUTPUT_DIR="${DEFAULT_EVAL_OUTPUT_DIR}"

# --- Activate Python Environment ---
# Activation (if any) is handled by util_env.sh via ACTIVATE_ENV

# --- Arrays to track checkpoints ---
SUBMITTED_CHECKPOINTS=()
SKIPPED_CHECKPOINTS=()

# --- Parse arguments ---
MODEL_PATHS=()
NUM_RECENT=""
CHECKPOINT_IDS=()
LORA_LOCAL_PATH=""
# TODO replace by your own SLURM config
SLURM_CONFIG="marenostrum"
SKIP_EXISTING="true"

while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            NUM_RECENT="$2"
            shift 2
            ;;
        -c)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^- ]]; do
                CHECKPOINT_IDS+=("$1")
                shift
            done
            ;;
        -l|--lora)
            LORA_LOCAL_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--slurm-config)
            SLURM_CONFIG="$2"
            shift 2
            ;;
        --no-skip-existing|--force)
            SKIP_EXISTING="false"
            shift
            ;;
        *)
            MODEL_PATHS+=("$1")
            shift
            ;;
    esac
done

# --- Error Check ---
if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "❌ ERROR: No model path was specified." >&2
    echo "Usage: run_chat_long_context.sh <model_path1> [model_path2 ...] [-n NUM_RECENT] [-c checkpoint1 checkpoint2 ...] [-l LORA_PATH] [-o OUTPUT_DIR]" >&2
    echo "Examples:" >&2
    echo "  run_chat_long_context.sh /path/to/model                       # Evaluate all checkpoints" >&2
    echo "  run_chat_long_context.sh /path/to/model1 /path/to/model2      # Evaluate all checkpoints for multiple models" >&2
    echo "  run_chat_long_context.sh /path/to/model -n 3                  # Evaluate 3 most recent" >&2
    echo "  run_chat_long_context.sh /path/to/model -c 1000 2000          # Evaluate specific checkpoints" >&2
    echo "  run_chat_long_context.sh /path/to/model/checkpoint-1000       # Evaluate single checkpoint" >&2
    echo "  run_chat_long_context.sh /path/to/model -o /tmp/my_outputs    # Set custom output directory" >&2
    echo "  run_chat_long_context.sh /path/to/model --no-skip-existing   # Force re-evaluation even if results exist" >&2
    exit 1
fi

# --- Validate LoRA Path if provided ---
if [ -n "$LORA_LOCAL_PATH" ]; then
    if ! is_lora_model "$LORA_LOCAL_PATH"; then
        echo "❌ ERROR: Provided LoRA path '$LORA_LOCAL_PATH' is invalid (adapter_model.safetensors not found)." >&2
        exit 1
    fi
fi

# Ensure OUTPUT_DIR exists
mkdir -p "$OUTPUT_DIR" || { echo "❌ ERROR: Cannot create output dir: $OUTPUT_DIR" >&2; exit 1; }

# Clean up any existing lock files in Hugging Face datasets cache
if [ -n "$HF_CACHE_REMOTE" ]; then
    CACHE_DATASETS_DIR="$HF_CACHE_REMOTE/huggingface/datasets"
    if [ -d "$CACHE_DATASETS_DIR" ]; then
        find "$CACHE_DATASETS_DIR" -type f -name "*.lock" -delete || true
    fi
fi


# --- Process each model path ---
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    checkpoints=()

    echo ""
    echo "=========================================="
    echo "Processing model: $MODEL_PATH"
    [ -n "$LORA_LOCAL_PATH" ] && echo "Using LoRA: $LORA_LOCAL_PATH"
    echo "Output directory: $OUTPUT_DIR"
    echo "=========================================="

    # Logic to handle Local Path vs Hugging Face Hub ID
    if [ -d "$MODEL_PATH" ]; then
        # Path exists locally, use existing logic
        if is_checkpoint_path "$MODEL_PATH"; then
            echo "Detected local checkpoint path: evaluating single checkpoint"
            checkpoints+=("$MODEL_PATH")
            if [[ "$MODEL_PATH" =~ checkpoint- ]]; then
                MODEL_PATH="$(dirname "$MODEL_PATH")"
            fi
        elif [ ${#CHECKPOINT_IDS[@]} -gt 0 ]; then
            for ckpt_id in "${CHECKPOINT_IDS[@]}"; do
                ckpt_path="$MODEL_PATH/checkpoint-$ckpt_id"
                [ -d "$ckpt_path" ] && checkpoints+=("$ckpt_path") || echo "⚠️ Warning: $ckpt_path not found"
            done
        else
            mapfile -t checkpoints < <(get_sorted_checkpoints "$MODEL_PATH")
            if [ -n "$NUM_RECENT" ]; then
                checkpoints=("${checkpoints[@]:0:$NUM_RECENT}")
            fi
        fi
    else
        # Path does NOT exist locally. Assume it's a Hugging Face Hub ID.
        echo "🌐 Local path not found. Treating '$MODEL_PATH' as a Hugging Face Hub ID."
        checkpoints+=("$MODEL_PATH")
    fi

    # Final check
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "❌ No valid model or checkpoints found for: $MODEL_PATH"
        exit 1
    fi

    echo "Found ${#checkpoints[@]} checkpoint(s) to evaluate"

    # --- Process each checkpoint ---
    for model in "${checkpoints[@]}"; do
        # Check if we can submit more jobs
        if ! can_submit_jobs; then
            wait_for_job_slots
        fi

        checkpoint_name=$(basename "$model")
        echo ""
        echo "------------------------------------------"
        echo "Evaluating checkpoint: $checkpoint_name"
        echo "------------------------------------------"

        # --- Determine output directory based on context ---
        if [[ "$OUTPUT_DIR" != "$DEFAULT_EVAL_OUTPUT_DIR" ]]; then
            # User provided -o flag: use OUTPUT_DIR directly, no subdirectories
            outdir="$OUTPUT_DIR"
        else
            # Default behavior: create subdirectories based on model/checkpoint structure
            model_name=$(basename "$MODEL_PATH")
            checkpoint_name=$(basename "$model")

            # If checkpoint_name equals model_name, it's a direct model (no checkpoints)
            if [ "$checkpoint_name" == "$model_name" ]; then
                outdir="$OUTPUT_DIR/$model_name"
            else
                # Model has checkpoints: create model/checkpoint structure
                outdir="$OUTPUT_DIR/$model_name/$checkpoint_name"
            fi
        fi

        # Check if output directory exists and has result files
        if should_skip_output_dir "$outdir" "$SKIP_EXISTING"; then
            echo "⏭️  Skipping already evaluated checkpoint: $checkpoint_name"
            SKIPPED_CHECKPOINTS+=("$checkpoint_name")
            continue
        fi

        mkdir -p "$outdir"

        # Construct lora argument
        lora_cmd_part=""
        if [ -n "$LORA_LOCAL_PATH" ]; then
            lora_cmd_part="--lora_local_path $LORA_LOCAL_PATH"
        fi

        # Long context tasks

        # babilong 0k
        echo "  Scheduling babilong 0k"
        amalia --tasks "babilong" \
            --model_name_or_path "$model" \
            --max_length 32768 \
            $lora_cmd_part \
            --model_output_dir "$outdir" \
            --slurm_config "$SLURM_CONFIG" \
            --mem "100GB" \
            --harness-args.apply_chat_template \
            --y

        babilong_lengths=("4k" "8k" "16k" "32k")
        for len in "${babilong_lengths[@]}"; do
          echo "  Scheduling babilong for length: $len"
          amalia --tasks "babilong_longctx" \
              --model_name_or_path "$model" \
              --max_length 32768 \
              $lora_cmd_part \
              --model_output_dir "$outdir" \
              --slurm_config "$SLURM_CONFIG" \
              --mem "100GB" \
              --harness-args.apply_chat_template \
              --harness-args.metadata "{\"max_seq_lengths\": \"$len\"}" \
              --y

          done

        # TODO to work needs a custom version of lm eval harness that correctly passes metadata to model
        # Ruler with various sequence lengths
        lengths=(4096 8192 16384 32768)
        # we paralelize the runs for different sequence lengths since they are independent and it is more efficient
        for len in "${lengths[@]}"; do
          echo "Scheduling ruler for length: $len"

          amalia --tasks "ruler_en" \
              --model_name_or_path "$model" \
              --max_length 32768 \
              $lora_cmd_part \
              --model_output_dir "$outdir" \
              --slurm_config "$SLURM_CONFIG" \
              --mem "100GB" \
              --harness-args.apply_chat_template \
              --harness-args.metadata '{"max_seq_lengths": ['"$len"']}' \
              --y
        done

        # Ruler pt-pt

        for len in "${lengths[@]}"; do
          echo "Scheduling ruler pt for length: $len"
          amalia --tasks "ruler_pt" \
              --model_name_or_path "$model" \
              --max_length 32768 \
              $lora_cmd_part \
              --model_output_dir "$outdir" \
              --slurm_config "$SLURM_CONFIG" \
              --mem "100GB" \
              --harness-args.apply_chat_template \
              --harness-args.metadata '{"max_seq_lengths": ['"$len"']}' \
              --y
        done

        echo "✅ Submitted jobs for checkpoint: $checkpoint_name"
        SUBMITTED_CHECKPOINTS+=("$checkpoint_name")
    done
done

echo ""
echo "=========================================="
echo "All evaluations scheduled!"
echo "=========================================="
echo ""
echo "📊 Summary:"
echo "  Total checkpoints found: ${#checkpoints[@]}"
echo "  Submitted: ${#SUBMITTED_CHECKPOINTS[@]}"
echo "  Skipped: ${#SKIPPED_CHECKPOINTS[@]}"
echo ""

if [ ${#SUBMITTED_CHECKPOINTS[@]} -gt 0 ]; then
    echo "✅ Submitted checkpoints:"
    for ckpt in "${SUBMITTED_CHECKPOINTS[@]}"; do
        echo "   - $ckpt"
    done
    echo ""
fi

if [ ${#SKIPPED_CHECKPOINTS[@]} -gt 0 ]; then
    echo "⏭️  Skipped checkpoints (already evaluated):"
    for ckpt in "${SKIPPED_CHECKPOINTS[@]}"; do
        echo "   - $ckpt"
    done
fi

