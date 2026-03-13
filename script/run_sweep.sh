#!/bin/bash
# run_sweep.sh — Hyperparameter sweep over all downstream tasks.
#
# Usage:
#   bash run_sweep.sh [OPTIONS]
#
# Options:
#   --model PATH          Path to encoder checkpoint (default: see MODEL below)
#   --tokenizer PATH      Path to tokenizer (default: see TOKENIZER below)
#   --tasks NAMES         Comma-separated list of tasks to run.
#                         Choices: fluorescence,stability,remote_homology,ss3,ss8,contact
#                         Default: all tasks
#   --lrs RATES           Comma-separated learning rates  (default: 1e-5,3e-5,1e-4)
#   --batch_sizes SIZES   Comma-separated per-device batch sizes (default: 2,4)
#   --epochs NUMS         Comma-separated epoch counts (default: 10,15,20)
#   --seed N              Random seed (default: 3)
#   --parallel            Run jobs in background (parallel). Default: sequential.
#   --dry_run             Print commands without executing.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="../outputs/glprotein_full/checkpoint-100000/encoder"
TOKENIZER="../outputs/glprotein_full/checkpoint-100000/protein_tokenizer"
TASKS="contact,ss3,ss8"
LRS="1e-5,1e-4,1e-3"
BATCH_SIZES="2,4"
EPOCHS="5,10,15"
SEED=3
PARALLEL=false
DRY_RUN=false

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";       shift 2 ;;
        --tokenizer)   TOKENIZER="$2";   shift 2 ;;
        --tasks)       TASKS="$2";       shift 2 ;;
        --lrs)         LRS="$2";         shift 2 ;;
        --batch_sizes) BATCH_SIZES="$2"; shift 2 ;;
        --epochs)      EPOCHS="$2";      shift 2 ;;
        --seed)        SEED="$2";        shift 2 ;;
        --parallel)    PARALLEL=true;    shift 1 ;;
        --dry_run)     DRY_RUN=true;     shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Task-specific fixed params ─────────────────────────────────────────────────
# Each line: OPTIMIZER FROZEN_BERT GRAD_ACCUM EVAL_BS EVAL_STEP WARMUP_RATIO
declare -A TASK_OPTIMIZER=(
    [fluorescence]="Adam"
    [stability]="AdamW"
    [remote_homology]="AdamW"
    [ss3]="AdamW"
    [ss8]="AdamW"
    [contact]="AdamW"
)
declare -A TASK_FROZEN_BERT=(
    [fluorescence]="True"
    [stability]="False"
    [remote_homology]="False"
    [ss3]="False"
    [ss8]="False"
    [contact]="False"
)
declare -A TASK_GRAD_ACCUM=(
    [fluorescence]=16
    [stability]=32
    [remote_homology]=16
    [ss3]=16
    [ss8]=32
    [contact]=8
)
declare -A TASK_EVAL_BS=(
    [fluorescence]=32
    [stability]=16
    [remote_homology]=8
    [ss3]=4
    [ss8]=4
    [contact]=1
)
declare -A TASK_EVAL_STEP=(
    [fluorescence]=50
    [stability]=500
    [remote_homology]=50
    [ss3]=50
    [ss8]=50
    [contact]=50
)
declare -A TASK_WARMUP=(
    [fluorescence]="0.0"
    [stability]="0.08"
    [remote_homology]="0.08"
    [ss3]="0.08"
    [ss8]="0.08"
    [contact]="0.08"
)

# ── Helpers ────────────────────────────────────────────────────────────────────
IFS=',' read -ra TASK_LIST    <<< "$TASKS"
IFS=',' read -ra LR_LIST      <<< "$LRS"
IFS=',' read -ra BS_LIST      <<< "$BATCH_SIZES"
IFS=',' read -ra EPOCH_LIST   <<< "$EPOCHS"

SWEEP_LOG="../outputs/sweep_results.tsv"
mkdir -p "../outputs"
if [[ ! -f "$SWEEP_LOG" ]]; then
    echo -e "task\tlr\tbatch_size\tepochs\toutput_dir" > "$SWEEP_LOG"
fi

PIDS=()
TOTAL=$(( ${#TASK_LIST[@]} * ${#LR_LIST[@]} * ${#BS_LIST[@]} * ${#EPOCH_LIST[@]} ))
COUNT=0

echo "========================================"
echo " Downstream Hyperparameter Sweep"
echo " Tasks:        ${TASK_LIST[*]}"
echo " LRs:          ${LR_LIST[*]}"
echo " Batch sizes:  ${BS_LIST[*]}"
echo " Epochs:       ${EPOCH_LIST[*]}"
echo " Total runs:   $TOTAL"
echo " Mode:         $([ "$PARALLEL" = true ] && echo parallel || echo sequential)"
echo " Dry run:      $DRY_RUN"
echo "========================================"

# ── Main sweep loop ────────────────────────────────────────────────────────────
for TASK in "${TASK_LIST[@]}"; do
    if [[ -z "${TASK_OPTIMIZER[$TASK]+x}" ]]; then
        echo "ERROR: Unknown task '$TASK'. Skipping."
        continue
    fi

    LOG_DIR="../outputs/${TASK}/sweep_logs"
    mkdir -p "$LOG_DIR"

    for LR in "${LR_LIST[@]}"; do
        for BS in "${BS_LIST[@]}"; do
            for EP in "${EPOCH_LIST[@]}"; do
                COUNT=$(( COUNT + 1 ))
                RUN_ID="lr${LR}_bs${BS}_ep${EP}"
                OUTPUT_FILE="${TASK}-sweep-${RUN_ID}"
                LOG_FILE="${LOG_DIR}/${RUN_ID}.out"

                CMD=(
                    bash ../run_main.sh
                    --model          "$MODEL"
                    --tokenizer_name "$TOKENIZER"
                    --output_file    "$OUTPUT_FILE"
                    --task_name      "$TASK"
                    --do_train       True
                    --epoch          "$EP"
                    --optimizer      "${TASK_OPTIMIZER[$TASK]}"
                    --per_device_batch_size        "$BS"
                    --gradient_accumulation_steps  "${TASK_GRAD_ACCUM[$TASK]}"
                    --eval_step      "${TASK_EVAL_STEP[$TASK]}"
                    --eval_batchsize "${TASK_EVAL_BS[$TASK]}"
                    --warmup_ratio   "${TASK_WARMUP[$TASK]}"
                    --learning_rate  "$LR"
                    --seed           "$SEED"
                    --frozen_bert    "${TASK_FROZEN_BERT[$TASK]}"
                    --delete_checkpoints_after_predict True
                )

                OUTPUT_DIR="../outputs/${TASK}/${SEED}-${OUTPUT_FILE}"
                echo -e "${TASK}\t${LR}\t${BS}\t${EP}\t${OUTPUT_DIR}" >> "$SWEEP_LOG"

                echo "[${COUNT}/${TOTAL}] ${TASK} | lr=${LR} bs=${BS} epochs=${EP}"

                # Skip if already completed
                if [[ -f "${OUTPUT_DIR}/prediction_done" ]]; then
                    echo "  Skipping — already complete."
                    continue
                fi

                # Detect state of this run directory
                MODEL_FILE=$(ls "${OUTPUT_DIR}/model.safetensors" "${OUTPUT_DIR}/pytorch_model.bin" 2>/dev/null | head -1 || true)
                LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)

                if [[ -n "$MODEL_FILE" ]]; then
                    # Training finished but prediction didn't complete — skip training
                    echo "  Model found, re-running prediction only."
                    CMD+=(--do_train False)
                elif [[ -n "$LATEST_CKPT" ]]; then
                    # Training was interrupted — resume from latest checkpoint
                    echo "  Resuming training from checkpoint: $(basename "$LATEST_CKPT")"
                    CMD+=(--resume_from_checkpoint "$LATEST_CKPT")
                fi

                if [[ "$DRY_RUN" = true ]]; then
                    echo "  CMD: ${CMD[*]} > ${LOG_FILE} 2>&1"
                    continue
                fi

                if [[ "$PARALLEL" = true ]]; then
                    "${CMD[@]}" > "$LOG_FILE" 2>&1 &
                    PIDS+=($!)
                    echo "  Launched PID $!"
                else
                    "${CMD[@]}" > "$LOG_FILE" 2>&1
                    echo "  Done. Log: $LOG_FILE"
                fi
            done
        done
    done
done

# ── Wait for parallel jobs ─────────────────────────────────────────────────────
if [[ "$PARALLEL" = true && ${#PIDS[@]} -gt 0 ]]; then
    echo ""
    echo "Waiting for ${#PIDS[@]} background jobs..."
    FAILED=0
    for PID in "${PIDS[@]}"; do
        if ! wait "$PID"; then
            echo "  WARNING: job PID $PID exited with error"
            FAILED=$(( FAILED + 1 ))
        fi
    done
    echo "All jobs finished. Failed: $FAILED"
fi

echo ""
echo "Sweep complete. Results index: $SWEEP_LOG"
