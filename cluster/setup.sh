#!/bin/bash
# ============================================================================
# Setup one-tantum per il cluster DMI — Feature-based Knowledge Distillation
#
# Uso (dal login node, dalla root del repo):
#   bash cluster/setup.sh
#
# Dipendenze: solo da configs/environment.yml (conda + sezione pip).
# Serve conda o mamba in PATH (es. module load …). Micromamba opzionale.
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

ENV_YML="$REPO_ROOT/configs/environment.yml"

echo "=== Setup Feature-based Knowledge Distillation (Cluster DMI) ==="
echo "Repo: $REPO_ROOT"
echo ""

if [ ! -f "$ENV_YML" ]; then
    echo "Manca $ENV_YML"
    exit 1
fi

CONDA_ENV_NAME="$(grep -E '^[[:space:]]*name:' "$ENV_YML" | head -1 | awk '{print $2}' | tr -d '\r')"
if [ -z "$CONDA_ENV_NAME" ]; then
    CONDA_ENV_NAME="dl-project"
fi

# True se la prima colonna di `conda info --envs` / `mamba env list` coincide col nome.
conda_env_table_has_name() {
    local list_cmd="$1" name="$2"
    $list_cmd 2>/dev/null | awk -v n="$name" '
        $1 == n { found = 1 }
        END { exit found ? 0 : 1 }
    '
}

# ── 1. Conda / mamba / micromamba ───────────────────────────────────────────
echo "Ambiente conda da: $ENV_YML (nome: $CONDA_ENV_NAME)"

if command -v micromamba &>/dev/null; then
    echo "Usando: micromamba"
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
    eval "$(micromamba shell hook -s bash)"
    if conda_env_table_has_name "micromamba env list" "$CONDA_ENV_NAME"; then
        echo "Aggiornamento ambiente esistente..."
        micromamba env update -f "$ENV_YML" -n "$CONDA_ENV_NAME" --prune
    else
        echo "Creazione ambiente..."
        micromamba create -f "$ENV_YML" -y
    fi
    micromamba activate "$CONDA_ENV_NAME"
elif command -v mamba &>/dev/null; then
    echo "Usando: mamba"
    if command -v conda &>/dev/null; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    eval "$(mamba shell hook --shell bash)"
    if conda_env_table_has_name "mamba env list" "$CONDA_ENV_NAME"; then
        echo "Aggiornamento ambiente esistente..."
        mamba env update -f "$ENV_YML" -n "$CONDA_ENV_NAME" --prune
    else
        echo "Creazione ambiente..."
        mamba env create -f "$ENV_YML"
    fi
    mamba activate "$CONDA_ENV_NAME"
elif command -v conda &>/dev/null; then
    echo "Usando: conda"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda_env_table_has_name "conda env list" "$CONDA_ENV_NAME"; then
        echo "Aggiornamento ambiente esistente..."
        conda env update -f "$ENV_YML" -n "$CONDA_ENV_NAME" --prune
    else
        echo "Creazione ambiente..."
        conda env create -f "$ENV_YML"
    fi
    conda activate "$CONDA_ENV_NAME"
else
    echo "Nessun di: micromamba, mamba, conda trovato in PATH."
    echo "Carica un modulo conda (es. module load miniforge3) oppure installa Micromamba, poi rilancia."
    exit 1
fi

PY="$(command -v python)"
if [ -z "$PY" ]; then
    PY="$(command -v python3)"
fi
if [ -z "$PY" ]; then
    echo "Python non trovato dopo l'attivazione dell'ambiente conda."
    exit 1
fi
echo "Python ambiente: $($PY --version 2>&1) ($PY)"
echo ""

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
") || { echo "Errore nel rilevamento GPU (torch nell'ambiente conda?)"; exit 1; }

echo "$GPU_INFO"

# ── 3. Asset offline (CIFAR-100 + pesi teacher) ──────────────────────────────
echo ""
# --soft: non fallisce il setup se mancano file (vedi cluster/check_offline_assets.sh)
bash "$REPO_ROOT/cluster/check_offline_assets.sh" --soft

# ── 4. Verifica installazione ─────────────────────────────────────────────────
echo ""
echo "Verifica installazione (pacchetti da environment.yml)..."
$PY -c "
from importlib.metadata import version as pkg_version
import torch
import torchvision
import yaml
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import tqdm
print(f'  PyTorch:       {torch.__version__}')
print(f'  Torchvision:   {torchvision.__version__}')
print(f'  PyYAML:        {yaml.__version__}')
print(f'  NumPy:         {np.__version__}')
print(f'  Pandas:        {pd.__version__}')
print(f'  Matplotlib:    {matplotlib.__version__}')
print(f'  scikit-learn:  {sklearn.__version__}')
print(f'  TensorBoard:   {pkg_version(\"tensorboard\")}')
print(f'  tqdm:          {tqdm.__version__}')
"

echo ""
echo "=== Setup completato ==="
echo "Ambiente conda: $CONDA_ENV_NAME — attivalo nei job con: conda activate $CONDA_ENV_NAME"
echo "(o mamba/micromamba activate, a seconda di cosa hai usato sopra)"
echo ""
echo "Prossimi passi:"
echo "  1. Controllo asset (salti scp se già tutto presente): bash cluster/check_offline_assets.sh"
echo "  2. Allinea cluster/train.sh (path repo, CONFIG=configs/... , Python dell'env)."
echo "  3. Esempio: CONFIG=configs/phase1_baseline.yaml sbatch cluster/train.sh"
