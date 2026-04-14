#!/bin/bash
# ============================================================================
# SLURM batch script — Evaluation sul cluster DMI
#
# Uso:
#   CONFIG=experiments/configs/nothink/curriculum/grpo_smollm2_360m.yaml sbatch cluster/eval.sh
#   CONFIG=experiments/configs/nothink/curriculum/grpo_smollm2_360m.yaml CURRICULUM=1 sbatch cluster/eval.sh
#   CONFIG=experiments/configs/nothink/standard/grpo_qwen05.yaml COMPARE=1 sbatch cluster/eval.sh
#   CONFIG=experiments/configs/baseline.yaml sbatch cluster/eval.sh
#   CHECKPOINT="path/to/ckpt" CONFIG=... sbatch cluster/eval.sh
#
# Il tipo di eval (grpo/baseline) viene auto-rilevato dal config.
# Se baseline results.json non esiste e COMPARE=1, lo script
# esegue anche la baseline evaluation automaticamente.
# ============================================================================

# ┌────────────────────────────────────────────────────────┐
# │  CONFIGURA QUI — modifica account/partition/qos/email  │
# └────────────────────────────────────────────────────────┘
#SBATCH --job-name=eval
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bellamacina50@gmail.com
#SBATCH --output=logs/slurm-eval-%j.log

# ── Variabili progetto ────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-}"  # vuoto = auto-detect, oppure path esplicito
COMPARE="${COMPARE:-0}"       # 0 = solo eval, 1 = anche comparison con baseline
CURRICULUM="${CURRICULUM:-0}" # 0 = singolo checkpoint, 1 = eval tutti gli stage (implica COMPARE=1)
SKIP_STAGES="${SKIP_STAGES:-0}" # numero di stage da saltare (per resume eval)

# Curriculum ha priorità su compare
if [ "$CURRICULUM" = "1" ]; then
    COMPARE="1"
fi

if [ -z "$CONFIG" ]; then
    echo "❌ CONFIG non impostato. Uso:"
    echo "  CONFIG=experiments/configs/nothink/curriculum/grpo_smollm2_360m.yaml sbatch cluster/eval.sh"
    echo ""
    echo "Config disponibili:"
    find experiments/configs -name 'grpo_*.yaml' -o -name 'baseline.yaml' 2>/dev/null | sort | sed 's/^/  /'
    exit 1
fi

# ── Setup ambiente ───────────────────────────────────────────────────────────
set -e

echo "============================================"
echo "  Evaluation — Cluster DMI"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG}"
echo "  Compare:   ${COMPARE}"
echo "  Curriculum: ${CURRICULUM}"
echo "  Checkpoint: ${CHECKPOINT:-auto}"
echo "============================================"

mkdir -p logs

cd "$HOME/GRPO-strict-generation"

# Genera dataset se mancante
if [ ! -d "data/synthetic" ]; then
    echo "Dataset non trovato, generazione in corso..."
    apptainer run --nv /shared/sifs/latest.sif \
        python -m src.datasets.synthetic_dataset --config "${CONFIG}"
fi

# Costruisci argomenti — il tipo di eval (grpo/baseline) viene auto-rilevato
# dal __main__.py in base al contenuto del config YAML.
EVAL_ARGS="--config ${CONFIG}"
if [ -n "$CHECKPOINT" ]; then
    EVAL_ARGS="${EVAL_ARGS} --checkpoint ${CHECKPOINT}"
fi
if [ "$COMPARE" = "1" ]; then
    EVAL_ARGS="${EVAL_ARGS} --compare"
fi
if [ "$CURRICULUM" = "1" ]; then
    EVAL_ARGS="${EVAL_ARGS} --curriculum"
fi
if [ "$SKIP_STAGES" != "0" ] && [ -n "$SKIP_STAGES" ]; then
    EVAL_ARGS="${EVAL_ARGS} --skip-stages ${SKIP_STAGES}"
fi

echo ""
echo "Avvio evaluation dentro Apptainer..."
echo "  Args: ${EVAL_ARGS}"
echo ""

# ── Esecuzione dentro container Apptainer ─────────────────────────────────────
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    /shared/sifs/latest.sif \
    python -m src.evaluation ${EVAL_ARGS}

echo ""
echo "============================================"
echo "  Evaluation completata!"
echo "  $(date)"
echo "============================================"
