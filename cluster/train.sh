#!/bin/bash
# ============================================================================
# SLURM batch script — Training (teacher/baseline) per questo repo
#
# Uso:
#   CONFIG=configs/teacher_finetune.yaml sbatch cluster/train.sh
#   CONFIG=configs/phase1_baseline.yaml sbatch cluster/train.sh
#
# Nota: la parte Apptainer è cluster-specific. Se non usi container, lo script
# esegue direttamente python3 sul nodo.
# ============================================================================

# ┌────────────────────────────────────────────────────────┐
# │  CONFIGURA QUI — modifica account/partition/qos/email  │
# └────────────────────────────────────────────────────────┘
#SBATCH --job-name=train
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bellamacina50@gmail.com
#SBATCH --output=experiments/logs/slurm-train-%j.log

# ── Variabili progetto ────────────────────────────────────────────────────────
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -z "$CONFIG" ]; then
    echo "❌ CONFIG non impostato. Uso:"
    echo "  CONFIG=experiments/configs/nothink/curriculum/grpo_smollm2_360m.yaml sbatch cluster/train.sh"
    echo ""
    echo "Config disponibili:"
    find experiments/configs -name 'grpo_*.yaml' -type f 2>/dev/null | sort | sed 's/^/  /'
    exit 1
fi

# ── Setup ambiente ───────────────────────────────────────────────────────────
set -e

echo "============================================"
echo "  Training — Cluster DMI"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG}"
echo "  Extra:     ${EXTRA_ARGS}"
echo "============================================"

# Crea directory logs se non esiste
mkdir -p experiments/logs

export WANDB_MODE=offline

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# (Opzionale) verifica asset offline (dataset/pesi) prima di partire
bash cluster/check_offline_assets.sh --soft || true

echo ""
echo "Avvio training..."
echo ""

# ── Esecuzione ────────────────────────────────────────────────────────────────
# Se vuoi usare un container, esporta APPTAINER_IMAGE=/path/to/image.sif
# TODO: imposta APPTAINER_IMAGE in base al tuo cluster (es. /shared/sifs/latest.sif).
if [ -n "${APPTAINER_IMAGE:-}" ]; then
    echo "Container: $APPTAINER_IMAGE"
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        "$APPTAINER_IMAGE" \
        bash -lc "python3 -u -c \"import sys; print(sys.version)\" && python3 -u -m pip --version && python3 -u -m src.training.train_teacher_finetune --config '${CONFIG}' ${EXTRA_ARGS}"
else
    # Heuristica: sceglie lo script in base al nome del config
    if [[ "$CONFIG" == *teacher* ]]; then
        python3 -u src/training/train_teacher_finetune.py --config "${CONFIG}" ${EXTRA_ARGS}
    else
        python3 -u src/training/train_baseline.py --config "${CONFIG}" ${EXTRA_ARGS}
    fi
fi

echo ""
echo "============================================"
echo "  Training completato!"
echo "  $(date)"
echo "============================================"
