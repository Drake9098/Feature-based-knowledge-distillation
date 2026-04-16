#!/bin/bash
# ============================================================================
# SLURM batch script — Training (teacher/baseline) per questo repo
#
# Uso:
#   bash cluster/submit_train.sh --config configs/teacher_finetune.yaml
#   bash cluster/submit_train.sh --config configs/phase1_baseline.yaml
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
#SBATCH --mail-user=salvatore.iurato1@studium.unict.it
# Nota: preferisci `cluster/submit_train.sh` per avere anche il tipo nel nome file.
#SBATCH --output=experiments/logs/slurm-%x-%j.log

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

export WANDB_MODE=offline

# SLURM esegue una copia dello script in una dir di spool: usa la submit dir come root progetto.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# Crea directory logs se non esiste (nel repo)
mkdir -p experiments/logs

# Rendi importabile `src/` senza installazione pacchetto
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Container di default (sul cluster DMI di solito esiste). Puoi sovrascrivere con APPTAINER_IMAGE=...
APPTAINER_IMAGE="${APPTAINER_IMAGE:-/shared/sifs/latest.sif}"

# (Opzionale) verifica asset offline (dataset/pesi) prima di partire
bash cluster/check_offline_assets.sh --soft || true

echo ""
echo "Avvio training..."
echo ""

# ── Esecuzione ────────────────────────────────────────────────────────────────
# Heuristica: sceglie lo script in base al nome del config
if [[ "$CONFIG" == *teacher* ]]; then
    ENTRYPOINT="src/training/train_teacher_finetune.py"
else
    ENTRYPOINT="src/training/train_baseline.py"
fi

# Se torch non è importabile sul nodo, usa Apptainer automaticamente.
if python3 -c "import torch" >/dev/null 2>&1; then
    echo "Python nodo: torch disponibile → eseguo nativamente."
    python3 -u "$ENTRYPOINT" --config "${CONFIG}" ${EXTRA_ARGS}
else
    echo "Python nodo: torch NON disponibile → eseguo in Apptainer: $APPTAINER_IMAGE"
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONPATH="$PYTHONPATH" \
        "$APPTAINER_IMAGE" \
        python3 -u "$ENTRYPOINT" --config "${CONFIG}" ${EXTRA_ARGS}
fi

echo ""
echo "============================================"
echo "  Training completato!"
echo "  $(date)"
echo "============================================"
