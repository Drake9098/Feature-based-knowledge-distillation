#!/bin/bash
# ============================================================================
# Watcher: controlla periodicamente la coda training e sottomette il prossimo job
# quando non ci sono job attivi dell'utente.
#
# Uso:
#   bash cluster/queue_watch.sh --poll 60
#
# Stop:
#   kill <PID>   (il PID viene scritto in .train_queue_pid)
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

POLL=60
while [ $# -gt 0 ]; do
  case "$1" in
    --poll) POLL="$2"; shift 2 ;;
    -h|--help)
      echo "Uso: bash cluster/queue_watch.sh [--poll SECONDI]"
      exit 0
      ;;
    *) echo "Argomento sconosciuto: $1"; exit 1 ;;
  esac
done

echo $$ > "$REPO_ROOT/.train_queue_pid"
echo "[queue] watcher start (pid $$) poll=${POLL}s"

trap 'rm -f "$REPO_ROOT/.train_queue_pid"' EXIT

while true; do
  bash "$REPO_ROOT/cluster/queue_next.sh" || true
  sleep "$POLL"
done

