#!/bin/bash
# ============================================================================
# Pulizia workspace per questo repo (CIFAR/KD)
#
# Uso:
#   bash cluster/clean.sh          # dry-run (mostra cosa cancellerebbe)
#   bash cluster/clean.sh --force  # cancella davvero
# ============================================================================

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

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

# ── Dataset (non cancelliamo di default) ─────────────────────────────────────
echo "[1/7] dataset/ (NON toccato di default)"
echo "      (se vuoi cancellarlo, fallo manualmente: rm -rf dataset/)"

# ── Checkpoints ──────────────────────────────────────────────────────────────
echo "[2/7] checkpoints/"
if [ -d "checkpoints" ]; then
    $CMD checkpoints/*
fi

# ── Logs SLURM / locali ─────────────────────────────────────────────────────
echo "[3/7] logs/"
if [ -d "logs" ]; then
    $CMD logs/*
fi

# ── runs/ (TensorBoard) ─────────────────────────────────────────────────────
echo "[4/7] runs/"
if [ -d "runs" ]; then
    $CMD runs/*
fi

# ── Cache Python ─────────────────────────────────────────────────────────
echo "[5/7] __pycache__/"
find . -type d -name "__pycache__" -print -exec $CMD {} + 2>/dev/null || true

# ── wandb offline runs (se presenti) ─────────────────────────────────────
echo "[6/7] wandb/"
if [ -d "wandb" ]; then
    $CMD wandb
fi

echo "[7/7] file di stato pipeline (nessuno in questo repo)"

echo ""
if [ "$FORCE" = "0" ]; then
    echo "=== Nessun file cancellato (dry-run). Usa: bash cluster/clean.sh --force ==="
else
    echo "Pulizia completata."
fi
