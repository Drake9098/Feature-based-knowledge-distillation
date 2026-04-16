#!/bin/bash
# ============================================================================
# Helper per sottomettere cluster/train.sh con un nome log descrittivo.
#
# Crea log Slurm in: experiments/logs/slurm-train-<type>-<jobid>.log
# dove <type> ∈ {teacher, baseline, student}.
#
# Uso:
#   bash cluster/submit_train.sh --config configs/teacher_finetune.yaml
#   bash cluster/submit_train.sh --config configs/phase1_baseline.yaml
#   CONFIG=configs/teacher_finetune.yaml bash cluster/submit_train.sh
#
# Note:
# - Il type viene derivato dal nome del config (heuristic).
# - Puoi forzare: TRAIN_TYPE=student bash cluster/submit_train.sh --config ...
# ============================================================================

set -euo pipefail

CONFIG_ARG="${CONFIG:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG_ARG="${2:-}"
            shift 2
            ;;
        *)
            echo "Argomento sconosciuto: $1"
            echo "Uso: bash cluster/submit_train.sh --config <path.yaml>"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG_ARG" ]; then
    echo "❌ Config mancante. Uso:"
    echo "  bash cluster/submit_train.sh --config configs/teacher_finetune.yaml"
    exit 1
fi

type="${TRAIN_TYPE:-}"
if [ -z "$type" ]; then
    cfg_lc="$(echo "$CONFIG_ARG" | tr '[:upper:]' '[:lower:]')"
    if [[ "$cfg_lc" == *teacher* ]]; then
        type="teacher"
    elif [[ "$cfg_lc" == *student* ]] || [[ "$cfg_lc" == *distill* ]] || [[ "$cfg_lc" == *kd* ]]; then
        type="student"
    else
        type="baseline"
    fi
fi

mkdir -p experiments/logs

job_name="train-${type}"
out_path="experiments/logs/slurm-train-${type}-%j.log"

echo "Submitting:"
echo "  CONFIG:   $CONFIG_ARG"
echo "  TYPE:     $type"
echo "  JOBNAME:  $job_name"
echo "  OUTPUT:   $out_path"

CONFIG="$CONFIG_ARG" sbatch --job-name="$job_name" --output="$out_path" cluster/train.sh

