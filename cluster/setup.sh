#!/bin/bash
# ============================================================================
# Setup one-tantum per il cluster DMI — Feature-based Knowledge Distillation
#
# Uso (dal login node, dalla root del repo):
#   bash cluster/setup.sh
#
# Dipendenze: pip install --user (in ~/.local) dal pyproject.toml in root.
# Non crea virtualenv.
#
# Lo script rilancia se stesso dentro srun + Apptainer automaticamente.
# ============================================================================

# ── 0. Auto-rilancio dentro srun + Apptainer se siamo sul login node ─────────
if [ -z "$APPTAINER_CONTAINER" ]; then
    echo "Login node rilevato → rilancio inside srun + Apptainer..."
    ACCOUNT="${SLURM_ACCOUNT:-dl-course-q2}"
    exec srun --account "$ACCOUNT" --partition "$ACCOUNT" --qos gpu-xlarge \
         --gres=gpu:1 --gres=shard:22000 --mem=48G --cpus-per-task=8 \
         apptainer run --nv /shared/sifs/latest.sif \
         bash "$0" "$@"
fi

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYPROJECT="$REPO_ROOT/pyproject.toml"

echo "=== Setup Feature-based Knowledge Distillation (Cluster DMI) ==="
echo "Repo: $REPO_ROOT"
echo ""

if [ ! -f "$PYPROJECT" ]; then
    echo "Manca $PYPROJECT"
    exit 1
fi

# ── 1. Python + pip ──────────────────────────────────────────────────────────
PY="$(command -v python3 || true)"
if [ -z "$PY" ]; then
    PY="$(command -v python || true)"
fi
if [ -z "$PY" ]; then
    echo "❌ Python non trovato in PATH."
    exit 1
fi
echo "Python: $($PY --version 2>&1) ($PY)"
echo ""

echo "📦 Aggiornamento pip..."
$PY -m pip install --user -U pip

echo "📦 Installazione progetto e dipendenze in ~/.local (pip --user)..."
$PY -m pip install --user -e .

echo ""
echo "NOTA: se il cluster non ha accesso a internet, pip potrebbe fallire."
echo "      In quel caso serve un wheelhouse offline o un modulo/container che già includa i pacchetti."
echo ""

# ── 2b. Directory esperimenti (logs/checkpoints) ──────────────────────────────
echo "Creazione directory output..."
mkdir -p "$REPO_ROOT/experiments/logs" "$REPO_ROOT/experiments/checkpoints"

# ── 2. Verifica GPU ──────────────────────────────────────────────────────────
echo "Rilevamento GPU..."

GPU_INFO=$($PY -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability()
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU: {name} (CC {cc[0]}.{cc[1]}, {vram:.1f} GB)')
else:
    print('  GPU: NESSUNA GPU rilevata')
") || { echo "Errore nel rilevamento GPU (torch installato e importabile?)"; exit 1; }

echo "$GPU_INFO"

# ── 3. Asset offline (CIFAR-100 + pesi teacher) ──────────────────────────────
echo ""
# --soft: non fallisce il setup se mancano file (vedi cluster/check_offline_assets.sh)
bash "$REPO_ROOT/cluster/check_offline_assets.sh" --soft

# ── 4. Verifica installazione ─────────────────────────────────────────────────
echo ""
echo "Verifica installazione (pip --user)..."
$PY -c "
import importlib
mods = ['torch', 'torchvision', 'yaml', 'tqdm']
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
print('  OK' if not missing else f'  Missing: {missing}')
"

echo ""
echo "=== Setup completato ==="
echo "Pip user-site: ~/.local (pip install --user)"
echo ""
echo "Prossimi passi:"
echo "  1. Controllo asset (salti scp se già tutto presente): bash cluster/check_offline_assets.sh"
echo "  2. Training teacher:  CONFIG=configs/teacher_finetune.yaml sbatch cluster/train.sh"
echo "  3. Training baseline: CONFIG=configs/phase1_baseline.yaml sbatch cluster/train.sh"
echo "  4. Log: experiments/logs/   Checkpoint: experiments/checkpoints/"
