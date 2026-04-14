#!/bin/bash
# ============================================================================
# Queue helper: sottomette il prossimo job SLURM dalla coda e riesegue se serve.
#
# Coda: file .train_queue nella root del repo, una riga = path YAML config.
# Questo script:
# - non è GRPO-specific
# - lavora 1-job-alla-volta (utile con QoS che limita a 1 GPU job)
#
# Uso:
#   bash cluster/queue_next.sh            # submit next if no active jobs
#   bash cluster/queue_next.sh --force    # submit next anche se hai job attivi (sconsigliato)
#
# Dipendenze: SLURM (squeue, sbatch), cluster/train.sh
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FORCE=0
if [ "${1:-}" = "--force" ]; then
  FORCE=1
fi

QUEUE_FILE="$REPO_ROOT/.train_queue"
STATE_FILE="$REPO_ROOT/.train_queue_state"

if [ ! -f "$QUEUE_FILE" ] || [ ! -s "$QUEUE_FILE" ]; then
  echo "[queue] Nessun job in coda ($QUEUE_FILE vuoto)."
  exit 0
fi

if [ "$FORCE" = "0" ]; then
  ACTIVE=$(squeue --me --noheader 2>/dev/null | wc -l | tr -d ' ')
  if [ "${ACTIVE:-0}" != "0" ]; then
    echo "[queue] Hai già job attivi ($ACTIVE). Non sottometto il prossimo."
    exit 0
  fi
fi

NEXT_CFG="$(head -n 1 "$QUEUE_FILE" | tr -d '\r')"
if [ -z "$NEXT_CFG" ]; then
  echo "[queue] Prima riga vuota in $QUEUE_FILE. Ripulisci il file."
  exit 1
fi

if [ ! -f "$REPO_ROOT/$NEXT_CFG" ] && [ ! -f "$NEXT_CFG" ]; then
  echo "[queue] Config non trovato: $NEXT_CFG"
  echo "[queue] Rimuovilo o correggilo in $QUEUE_FILE."
  exit 1
fi

echo "[queue] Sottometto: $NEXT_CFG"
JOBID=$(cd "$REPO_ROOT" && CONFIG="$NEXT_CFG" sbatch cluster/train.sh | awk '{print $4}')
echo "[queue] Job ID: $JOBID"
echo "$JOBID:$NEXT_CFG:$(date -Iseconds)" >> "$STATE_FILE"

# Pop first line
tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"

echo "[queue] Rimasti in coda: $(wc -l < "$QUEUE_FILE" | tr -d ' ')"
