#!/bin/bash
# ============================================================================
# Pulizia workspace sul cluster DMI
#
# Uso:
#   bash cluster/clean.sh          # dry-run (mostra cosa cancellerebbe)
#   bash cluster/clean.sh --force  # cancella davvero
# ============================================================================

set -e
cd "$HOME/GRPO-strict-generation"

FORCE=0
if [ "$1" = "--force" ]; then
    FORCE=1
fi

if [ "$FORCE" = "0" ]; then
    echo "=== DRY RUN — aggiungi --force per cancellare davvero ==="
    echo ""
    CMD="echo [DRY] rm -rf"
else
    CMD="rm -rf"
fi

echo "Pulizia workspace: $PWD"
echo ""

# ── Svuota data/ (dataset generato, verrà ricreato da train.sh) ───────────
echo "[1/9] data/ (dataset sintetico)"
if [ -d "data" ]; then
    $CMD data/*
fi

# ── Svuota experiments/ tranne configs/ ───────────────────────────────────
echo "[2/9] experiments/ (checkpoints, logs — preserva configs/)"
if [ -d "experiments/checkpoints" ]; then
    $CMD experiments/checkpoints/*
fi
if [ -d "experiments/logs" ]; then
    $CMD experiments/logs/*
fi

# ── Svuota logs/ (SLURM log) ─────────────────────────────────────────────
echo "[3/9] logs/ (SLURM output)"
if [ -d "logs" ]; then
    $CMD logs/*
fi

# ── Cache Python ─────────────────────────────────────────────────────────
echo "[4/9] __pycache__/"
find . -type d -name "__pycache__" -print -exec $CMD {} + 2>/dev/null || true

# ── Artifact LoRA/Unsloth del GRPOTrainer ────────────────────────────────
echo "[5/9] grpo_trainer_lora_model_*/"
for d in grpo_trainer_lora_model_*; do
    [ -d "$d" ] && $CMD "$d"
done

# ── wandb offline runs ──────────────────────────────────────────────────
echo "[6/9] wandb/ (cartella legacy)"
if [ -d "wandb" ]; then
    $CMD wandb
fi

# ── Unsloth compiled cache ──────────────────────────────────────────────
echo "[7/9] unsloth_compiled_cache/"
if [ -d "unsloth_compiled_cache" ]; then
    $CMD unsloth_compiled_cache
fi

# ── Notebooks (non servono sul cluster) ──────────────────────────────────
echo "[8/9] notebooks/"
if [ -d "notebooks" ]; then
    $CMD notebooks
fi

# ── File watcher / pipeline ──────────────────────────────────────────────
echo "[9/9] .job_chain, .chain_pid, .chain_failed, .chain_stopped, .monitor_cache"
[ -f ".job_chain" ] && $CMD .job_chain
[ -f ".chain_pid" ] && $CMD .chain_pid
[ -f ".chain_failed" ] && $CMD .chain_failed
[ -f ".chain_stopped" ] && $CMD .chain_stopped
[ -f ".monitor_cache" ] && $CMD .monitor_cache

echo ""
if [ "$FORCE" = "0" ]; then
    echo "=== Nessun file cancellato (dry-run). Usa: bash cluster/clean.sh --force ==="
else
    echo "Pulizia completata."
fi
