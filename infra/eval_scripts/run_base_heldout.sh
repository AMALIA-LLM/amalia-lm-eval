#!/bin/bash

umask 002

# Source environment helper to load .env and activate Python env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/util_env.sh"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/util_functions.sh"

# --- Output Directory Configuration ---
# Require DEFAULT_EVAL_OUTPUT_DIR to be set in .env unless user passes -o
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
MODEL_PATH=""
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
            MODEL_PATH="$1"
            shift
            ;;
    esac
done

# --- Error Check ---
if [ -z "$MODEL_PATH" ]; then
    echo "❌ ERROR: No model path was specified." >&2
    echo "Usage: run_base_heldout.sh <model_path> [-n NUM_RECENT] [-c checkpoint1 checkpoint2 ...] [-l LORA_PATH] [-o OUTPUT_DIR]" >&2
    echo "Examples:" >&2
    echo "  run_base_heldout.sh /path/to/model                       # Evaluate all checkpoints" >&2
    echo "  run_base_heldout.sh /path/to/model -n 3                  # Evaluate 3 most recent" >&2
    echo "  run_base_heldout.sh /path/to/model -c 1000 2000          # Evaluate specific checkpoints" >&2
    echo "  run_base_heldout.sh /path/to/model/checkpoint-1000       # Evaluate single checkpoint" >&2
    echo "  run_base_heldout.sh /path/to/model -o /tmp/my_outputs    # Set custom output directory" >&2
    echo "  run_base_heldout.sh /path/to/model --no-skip-existing   # Force re-evaluation even if results exist" >&2
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

# HELD OUT EVALUATION TASKS ONLY
tasks=( \
    "piqa-mt-pt" \
    "piqa" \
    "social_iqa-mt-pt" \
    "social_iqa" \
    "amalia_dsl_tl_pt_mcq_shuffled_no_na" \
    "openbookqa-mt-pt" \
    "openbookqa" \
    "winogrande" \
    "winogrande-mt-pt" \
)

# Concatenate tasks into a single string
tasks_string=""
for t in "${tasks[@]}"; do
    tasks_string="$tasks_string $t"
done

echo ""
echo "=========================================="
echo "Processing model (HELD-OUT ONLY): $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Check if the provided path is already a checkpoint
if is_checkpoint_path "$MODEL_PATH"; then
    echo "Detected checkpoint path: evaluating single checkpoint"
    if [ -d "$MODEL_PATH" ]; then
        checkpoints+=("$MODEL_PATH")
        # Update MODEL_PATH to parent directory for output naming
        # If path contains checkpoint-, extract parent; otherwise keep as is
        if [[ "$MODEL_PATH" =~ checkpoint- ]]; then
            MODEL_PATH="$(dirname "$MODEL_PATH")"
        fi
    else
        echo "❌ Checkpoint path does not exist: $MODEL_PATH"
        exit 1
    fi
elif [ ${#CHECKPOINT_IDS[@]} -gt 0 ]; then
    # Specific checkpoint IDs provided
    for ckpt_id in "${CHECKPOINT_IDS[@]}"; do
        ckpt_path="$MODEL_PATH/checkpoint-$ckpt_id"
        if [ -d "$ckpt_path" ]; then
            checkpoints+=("$ckpt_path")
        else
            echo "⚠️  Warning: Checkpoint $ckpt_path not found"
        fi
    done
else
    # Get all checkpoints sorted by number (newest first)
    mapfile -t checkpoints < <(get_sorted_checkpoints "$MODEL_PATH")

    # Limit to N most recent if specified
    if [ -n "$NUM_RECENT" ]; then
        checkpoints=("${checkpoints[@]:0:$NUM_RECENT}")
    fi
fi

# Check if we found any checkpoints
if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "❌ No checkpoints found for model: $MODEL_PATH"
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
    echo "Evaluating checkpoint (HELD-OUT): $checkpoint_name"
    echo "------------------------------------------"

    # --- Determine output directory based on context ---
    if [[ "$OUTPUT_DIR" != "${DEFAULT_EVAL_OUTPUT_DIR}" ]]; then
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

    # --- Base model held-out evaluation (5-shot, no chat template) ---
    echo "  Scheduling held-out base model tasks (5-shot): $tasks_string"
    amalia --tasks $tasks_string \
        --model_name_or_path "$model" \
        $lora_cmd_part \
        --model_output_dir "$outdir" \
        --slurm_config "$SLURM_CONFIG" \
        --harness-args.num_fewshot 5 \
        --y

    # "alba_mcq" and "alba_two_options_mcq" uses 3 shots
    amalia --tasks "alba_mcq" \
        --model_name_or_path $model \
        $lora_cmd_part \
        --model_output_dir $outdir \
        --slurm_config "$SLURM_CONFIG" \
        --harness-args.num_fewshot 3 \
        --y

    amalia --tasks "alba_two_options_mcq" \
        --model_name_or_path $model \
        $lora_cmd_part \
        --model_output_dir $outdir \
        --slurm_config "$SLURM_CONFIG" \
        --harness-args.num_fewshot 3 \
        --y

    # amalia_ptbrvid_mcq_shuffle is too long so it uses less shots
    amalia --tasks "amalia_ptbrvid_mcq_shuffle" \
        --model_name_or_path $model \
        $lora_cmd_part \
        --model_output_dir $outdir \
        --slurm_config "$SLURM_CONFIG" \
        --harness-args.num_fewshot 2 \
        --y

    # BBH (3 shots) - held-out
    amalia --tasks "bbh_cot_fewshot_mt_pt" \
        --model_name_or_path "$model" \
        $lora_cmd_part \
        --model_output_dir "$outdir" \
         --slurm_config "$SLURM_CONFIG" \
         --harness-args.num_fewshot 3 \
         --y

    amalia --tasks "bbh_cot_fewshot" \
        --model_name_or_path "$model" \
        $lora_cmd_part \
        --model_output_dir "$outdir" \
         --slurm_config "$SLURM_CONFIG" \
         --harness-args.num_fewshot 3 \
         --y

    echo "✅ Submitted jobs for checkpoint: $checkpoint_name"
    SUBMITTED_CHECKPOINTS+=("$checkpoint_name")
done

echo ""
echo "=========================================="
echo "All held-out evaluations scheduled!"
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