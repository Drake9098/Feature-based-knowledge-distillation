#!/bin/bash
# ============================================================================
# Verifica presenza di dataset CIFAR-100 (layout torchvision) e pesi teacher.
#
# Uso (dalla root del repo o da qualsiasi directory, lo script trova il repo):
#   bash cluster/check_offline_assets.sh
#   CONFIG=configs/teacher_finetune.yaml bash cluster/check_offline_assets.sh
#
# Opzioni:
#   --soft   non uscire con codice 1 se manca qualcosa (utile dentro setup.sh)
#
# Exit code: 0 se dataset e pesi sono presenti; 1 se manca almeno uno (salvo --soft).
#
# Se tutto è OK, non serve ricopiare dataset/ e weights/ con scp.
# ============================================================================

set -euo pipefail

SOFT=0
for arg in "$@"; do
    case "$arg" in
        --soft) SOFT=1 ;;
        -h|--help)
            grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -n "${CONFIG:-}" ]; then
    YAML_CFG="$CONFIG"
    [ -f "$YAML_CFG" ] || YAML_CFG="$REPO_ROOT/$CONFIG"
else
    YAML_CFG="$REPO_ROOT/configs/phase1_baseline.yaml"
fi

resolve_paths_with_python() {
    local cfg_path="$1"
    REPO_ROOT="$REPO_ROOT" CFG_PATH="$cfg_path" python3 <<'PY'
import os
import pathlib
import sys

try:
    import yaml
except ImportError:
    sys.exit(3)

repo = pathlib.Path(os.environ["REPO_ROOT"]).resolve()
cfg_path = pathlib.Path(os.environ["CFG_PATH"])
if not cfg_path.is_file():
    sys.exit(4)
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
data_root = (repo / pathlib.Path(cfg["data"]["root"])).resolve()
weights = (repo / pathlib.Path(cfg["model"]["teacher_weights_path"])).resolve()
cifar = data_root / "cifar-100-python"
print(str(cifar))
print(str(weights))
PY
}

CIFAR_MARKER=""
WEIGHTS_FILE=""

if [ -f "$YAML_CFG" ]; then
    if OUT=$(resolve_paths_with_python "$YAML_CFG" 2>/dev/null); then
        CIFAR_MARKER=$(printf '%s\n' "$OUT" | sed -n '1p')
        WEIGHTS_FILE=$(printf '%s\n' "$OUT" | sed -n '2p')
    fi
fi

if [ -z "$CIFAR_MARKER" ] || [ -z "$WEIGHTS_FILE" ]; then
    DATA_DIR="${DATA_DIR:-$REPO_ROOT/dataset}"
    CIFAR_MARKER="$DATA_DIR/cifar-100-python"
    WEIGHTS_FILE="${WEIGHTS_FILE:-$REPO_ROOT/weights/resnet50_imagenet1k_v1.pth}"
    if [ -f "$YAML_CFG" ] && ! python3 -c "import yaml" 2>/dev/null; then
        echo "Nota: PyYAML non disponibile; uso path di default (come in download_offline_assets.py)."
    fi
fi

echo "=== Controllo asset offline ==="
echo "Repo: $REPO_ROOT"
if [ -n "${CONFIG:-}" ]; then
    echo "YAML (da CONFIG): $CONFIG"
else
    echo "YAML (default):   $YAML_CFG"
fi
echo ""

DATA_OK=0
WEIGHTS_OK=0
[ -d "$CIFAR_MARKER" ] && DATA_OK=1
[ -f "$WEIGHTS_FILE" ] && WEIGHTS_OK=1

if [ "$DATA_OK" -eq 1 ]; then
    echo "[OK] Dataset CIFAR-100 (cartella torchvision): $CIFAR_MARKER"
else
    echo "[MANCANTE] Dataset CIFAR-100: attesa directory $CIFAR_MARKER"
fi

if [ "$WEIGHTS_OK" -eq 1 ]; then
    echo "[OK] Pesi teacher: $WEIGHTS_FILE"
else
    echo "[MANCANTE] Pesi teacher: atteso file $WEIGHTS_FILE"
fi

echo ""

if [ "$DATA_OK" -eq 1 ] && [ "$WEIGHTS_OK" -eq 1 ]; then
    echo "Tutto presente: non è necessario ricopiare dataset/ o weights/ con scp."
    exit 0
fi

echo "Per preparare gli asset su una macchina con rete, dalla root del repo:"
echo "  python scripts/download_offline_assets.py --dataset-dir ./dataset --weights-dir ./weights"
echo "Poi, solo se mancavano file, copia sul cluster con scp -r dataset weights"
echo ""

if [ "$SOFT" -eq 1 ]; then
    exit 0
fi
exit 1
