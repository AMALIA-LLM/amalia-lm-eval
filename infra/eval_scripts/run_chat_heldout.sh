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
    echo "Usage: run_chat_heldout.sh <model_path1> [model_path2 ...] [-n NUM_RECENT] [-c checkpoint1 checkpoint2 ...] [-l LORA_PATH] [-o OUTPUT_DIR]" >&2
    echo "Examples:" >&2
    echo "  run_chat_heldout.sh /path/to/model                       # Evaluate all checkpoints" >&2
    echo "  run_chat_heldout.sh /path/to/model1 /path/to/model2      # Evaluate all checkpoints for multiple models" >&2
    echo "  run_chat_heldout.sh /path/to/model -n 3                  # Evaluate 3 most recent" >&2
    echo "  run_chat_heldout.sh /path/to/model -c 1000 2000          # Evaluate specific checkpoints" >&2
    echo "  run_chat_heldout.sh /path/to/model/checkpoint-1000       # Evaluate single checkpoint" >&2
    echo "  run_chat_heldout.sh /path/to/model -o /tmp/my_outputs    # Set custom output directory" >&2
    echo "  run_chat_heldout.sh /path/to/model --no-skip-existing   # Force re-evaluation even if results exist" >&2
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

# Zero shot tasks
tasks=( \
    "piqa-mt-pt" \
    "piqa" \
    "social_iqa-mt-pt" \
    "social_iqa" \

    # other datasets
    "amalia_dsl_tl_pt_mcq_shuffled_no_na_chat" \
    "amalia_legalbench_pt_log" \
    "alba_mcq" \
    "alba_two_options_mcq" \
    "openbookqa" \
    "openbookqa-mt-pt" \
    "winogrande" \
    "winogrande-mt-pt" \

)

# Concatenate tasks into a single string
tasks_string=""
for t in "${tasks[@]}"; do
    tasks_string="$tasks_string $t"
done

# --- Process each model path ---
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    checkpoints=()

    echo ""
    echo "=========================================="
    echo "Processing model: $MODEL_PATH"
    [ -n "$LORA_LOCAL_PATH" ] && echo "Using LoRA: $LORA_LOCAL_PATH"
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

        # Zero shot tasks (num_fewshot 0)
        echo "  Scheduling Zero-Shot tasks: $tasks_string"
        amalia --tasks $tasks_string \
            --model_name_or_path "$model" \
            $lora_cmd_part \
            --model_output_dir "$outdir" \
            --slurm_config "$SLURM_CONFIG" \
            --harness-args.apply_chat_template \
            --harness-args.fewshot_as_multiturn \
            --harness-args.num_fewshot 0 \
            --y

        # same for amalia_ptbrvid_mcq_shuffle_chat avoid CUDA out of memory errors
        echo "Scheduling amalia_ptbrvid_mcq_shuffle_chat"
        amalia --tasks "amalia_ptbrvid_mcq_shuffle_chat" \
            --model_name_or_path "$model" \
            $lora_cmd_part \
            --model_output_dir "$outdir" \
            --slurm_config "$SLURM_CONFIG" \
            --harness-args.apply_chat_template \
            --harness-args.fewshot_as_multiturn \
            --harness-args.num_fewshot 0 \
            --harness-args.batch_size 1 \
            --y

        # BBH (3 shots)
        echo "Scheduling bbh_cot_fewshot_mt_pt (3 shots)"
        amalia --tasks "bbh_cot_fewshot_mt_pt" \
            --model_name_or_path "$model" \
            $lora_cmd_part \
            --model_output_dir "$outdir" \
             --slurm_config "$SLURM_CONFIG" \
             --harness-args.apply_chat_template \
             --harness-args.fewshot_as_multiturn \
             --harness-args.num_fewshot 3 \
             --y

        # BBH (3 shots)
        echo "Scheduling bbh_cot_fewshot (3 shots)"
        amalia --tasks "bbh_cot_fewshot" \
            --model_name_or_path "$model" \
            $lora_cmd_part \
            --model_output_dir "$outdir" \
             --slurm_config "$SLURM_CONFIG" \
             --harness-args.apply_chat_template \
             --harness-args.fewshot_as_multiturn \
             --harness-args.num_fewshot 3 \
             --y

        # other math datasets
        echo "  Scheduling aime math tasks"
        amalia --tasks "aime aime24 aime25 aime_pt aime24_pt aime25_pt" \
            --model_name_or_path "$model" \
            $lora_cmd_part \
            --model_output_dir "$outdir" \
            --slurm_config "$SLURM_CONFIG" \
            --harness-args.apply_chat_template \
            --harness-args.fewshot_as_multiturn \
            --harness-args.gen_kwargs "max_gen_toks=3200" \
            --y

        echo "  Scheduling minerva MATH tasks"
        amalia --tasks "minerva_math_en minerva_math500_en minerva_math_pt minerva_math500_pt" \
            --model_name_or_path "$model" \
            $lora_cmd_part \
            --model_output_dir "$outdir" \
            --slurm_config "$SLURM_CONFIG" \
            --harness-args.apply_chat_template \
            --harness-args.fewshot_as_multiturn \
            --harness-args.gen_kwargs "max_gen_toks=2048" \
            --y

        # math-pt mcq adapted to eval-harness
        amalia --tasks "math_pt_br_mcq math_pt_pt_mcq_no_figures" \
            --model_name_or_path "$model" \
            $lora_cmd_part \
            --model_output_dir "$outdir" \
            --slurm_config "$SLURM_CONFIG" \
            --harness-args.apply_chat_template \
            --harness-args.fewshot_as_multiturn \
            --y

        # pt culture coverage
        echo "  Scheduling Cultura Viva PT evaluation..."
        amalia --tasks "cultura_viva_pt_chat cultura_viva_pt_chat_cot" \
            --model_name_or_path "$model" \
            $lora_cmd_part \
            --model_output_dir "$outdir" \
            --slurm_config "$SLURM_CONFIG" \
            --harness-args.apply_chat_template \
            --y

        echo "  Scheduling temp_reasoning_pt evaluation..."
        amalia --tasks "saudade_chat_pt saudade_chat_cot_pt" \
            --model_name_or_path "$model" \
            $lora_cmd_part \
            --model_output_dir "$outdir" \
            --slurm_config "$SLURM_CONFIG" \
            --harness-args.apply_chat_template \
            --y

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

