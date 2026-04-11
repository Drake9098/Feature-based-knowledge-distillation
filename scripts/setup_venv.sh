#!/usr/bin/env bash
# Crea .venv nella root del repository e installa requirements.txt.
#
# Uso (cluster Linux):
#   chmod +x scripts/setup_venv.sh   # una tantum
#   ./scripts/setup_venv.sh          # crea / aggiorna dipendenze
#   source .venv/bin/activate        # attiva nella tua sessione SSH
#
# In alternativa, in una sola shell bash:
#   source scripts/setup_venv.sh     # crea (se manca), installa e attiva

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Errore: comando '$PYTHON' non trovato. Imposta PYTHON=/percorso/python se necessario." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creazione virtualenv in $ROOT/.venv ..."
  "$PYTHON" -m venv .venv
fi

echo "Aggiornamento pip e installazione dipendenze..."
./.venv/bin/pip install -U pip
./.venv/bin/pip install -r requirements.txt

# Se lo script è stato SOURCED, attiva il venv nella shell corrente.
# Se eseguito come ./scripts/setup_venv.sh, l'attivazione qui non persisterebbe: usa source .venv/bin/activate
if [[ "${BASH_SOURCE[0]:-}" != "${0:-}" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
  echo "Virtualenv attivato (shell corrente): $ROOT/.venv"
else
  echo "Fatto. Attiva con: source $ROOT/.venv/bin/activate"
fi
