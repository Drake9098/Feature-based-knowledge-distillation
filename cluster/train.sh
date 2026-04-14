#!/bin/bash
# ============================================================================
# SLURM batch script — Training sul cluster DMI
#
# Uso:
#   CONFIG=experiments/configs/nothink/curriculum/grpo_smollm2_360m.yaml sbatch cluster/train.sh
#   CONFIG=experiments/configs/nothink/standard/grpo_qwen05.yaml EXTRA_ARGS="--resume" sbatch cluster/train.sh
#
# Per il primo avvio eseguire prima:  bash cluster/setup.sh
# (dentro una sessione interattiva Apptainer)
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
#SBATCH --output=logs/slurm-train-%j.log

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
mkdir -p logs

# wandb in modalità offline (non c'è internet sul cluster per studenti)
# I dottorandi con accesso a internet possono commentare questa riga
export WANDB_MODE=offline

# Percorso progetto
cd "$HOME/GRPO-strict-generation"

# Genera dataset base solo se necessario (non-curriculum mode).
# In curriculum mode i dataset di training vengono generati dal codice Python
# per ogni stage, e l'eval dataset bilanciato viene creato automaticamente.
CURRICULUM_ENABLED=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG}'))
c = cfg.get('curriculum', {})
print('1' if c and c.get('enabled', False) else '0')
" 2>/dev/null || echo "0")

if [ "$CURRICULUM_ENABLED" = "0" ] && [ ! -d "data/synthetic/train" ]; then
    echo "Dataset non trovato, generazione in corso..."
    apptainer run --nv /shared/sifs/latest.sif \
        python -m src.datasets.synthetic_dataset --config "${CONFIG}"
elif [ "$CURRICULUM_ENABLED" = "1" ]; then
    echo "Curriculum mode: dataset generati automaticamente dal training"
fi

echo ""
echo "Avvio training dentro Apptainer..."
echo ""

# ── Esecuzione dentro container Apptainer ─────────────────────────────────────
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    /shared/sifs/latest.sif \
    python -m src.training --config "${CONFIG}" ${EXTRA_ARGS}

echo ""
echo "============================================"
echo "  Training completato!"
echo "  $(date)"
echo "============================================"
