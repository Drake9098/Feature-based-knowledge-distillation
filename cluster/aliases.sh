#!/bin/bash
# ============================================================================
# Alias utili per il cluster DMI
#
# Uso:
#   source cluster/aliases.sh
#
# Per caricarli automaticamente, aggiungi al tuo ~/.bashrc:
#   source ~/GRPO-strict-generation/cluster/aliases.sh
# ============================================================================

PROJ_DIR="$HOME/GRPO-strict-generation"

# ── Job management ───────────────────────────────────────────────────────────

# Controlla i miei job attivi
alias myjobs='squeue --me --format="%.10i %.20j %.8T %.10M %.6D %.20R %o"'

# Info dettagliata su un job (uso: jobinfo <JOB_ID>)
jobinfo() {
    if [ -z "$1" ]; then
        echo "Uso: jobinfo <JOB_ID>"
        return 1
    fi
    scontrol show job "$1"
}

# Cancella un job (uso: killjob <JOB_ID>)
alias killjob='scancel'

# Cancella tutti i miei job
alias killalljobs='watcher-kill && scancel --me'

# ── Log monitoring ───────────────────────────────────────────────────────────

# Segui il log di un job di training (uso: trainlog <JOB_ID>)
trainlog() {
    if [ -z "$1" ]; then
        echo "Uso: trainlog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-train-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Segui il log di un job di eval GRPO (uso: evallog <JOB_ID>)
evallog() {
    if [ -z "$1" ]; then
        echo "Uso: evallog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-eval-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Segui il log di un job baseline (uso: baselog <JOB_ID>)
baselog() {
    if [ -z "$1" ]; then
        echo "Uso: baselog <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-baseline-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile"
}

# Mostra l'ultimo log (qualsiasi tipo) — uso: lastlog [N_RIGHE]
# Senza argomento: tail -f (segui). Con argomento: mostra ultime N righe.
lastlog() {
    local logfile
    logfile=$(ls -t "$PROJ_DIR"/logs/slurm*.log 2>/dev/null | head -1)
    if [ -z "$logfile" ]; then
        echo "Nessun log trovato in $PROJ_DIR/logs/"
        return 1
    fi
    echo "==> $logfile <=="
    if [ -n "$1" ]; then
        tail -n "$1" "$logfile"
    else
        tail -f "$logfile"
    fi
}

# ── Filesystem ───────────────────────────────────────────────────────────────

# Tree ricorsivo di una cartella (uso: tree <DIR> [DEPTH])
tree() {
    local dir="${1:-.}"
    local depth="${2:-3}"
    find "$dir" -maxdepth "$depth" | sed -e "s|[^/]*/|  |g" -e "s|  |├─|"
}

# Alias più leggibile basato su ls -R (uso: ltree <DIR>)
ltree() {
    local dir="${1:-.}"
    ls -R "$dir" | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//──/g' -e 's/^/  /' -e 's/─/│/'
}

# ── GPU & risorse ────────────────────────────────────────────────────────────

# Stato GPU — esegue nvidia-smi sul nodo del job attivo via srun --overlap
gpu() {
    local jobid
    jobid=$(squeue --me --noheader --format="%i" 2>/dev/null | head -1)
    if [ -z "$jobid" ]; then
        echo "❌ Nessun job SLURM attivo."
        return 1
    fi
    srun --jobid="$jobid" --overlap nvidia-smi
}

# Uso disco del progetto
alias quota='quota -s'

# ── Quick commands ───────────────────────────────────────────────────────────

# Vai alla directory del progetto
alias proj='cd "$PROJ_DIR"'

# Mostra i checkpoint disponibili
# Uso: ckpts --think --curriculum
#       ckpts --nothink --all
#       ckpts --all
ckpts() {
    local flag_think=0 flag_nothink=0 flag_curriculum=0 flag_standard=0 flag_all=0
    for arg in "$@"; do
        case "$arg" in
            --think)      flag_think=1 ;;
            --nothink)    flag_nothink=1 ;;
            --curriculum) flag_curriculum=1 ;;
            --standard)   flag_standard=1 ;;
            --all)        flag_all=1 ;;
            --help|-h)
                echo "Uso: ckpts <VARIANTE>"
                echo "  --think/--nothink + --curriculum/--standard, oppure --all"
                return 0 ;;
            *) echo "❌ Argomento sconosciuto: $arg"; return 1 ;;
        esac
    done
    # Validazione
    if [ "$flag_think" -eq 1 ] && [ "$flag_nothink" -eq 1 ]; then
        echo "❌ --think e --nothink sono mutualmente esclusivi."; return 1
    fi
    if [ "$flag_curriculum" -eq 1 ] && [ "$flag_standard" -eq 1 ]; then
        echo "❌ --curriculum e --standard sono mutualmente esclusivi."; return 1
    fi
    local think_set=() curric_set=()
    if [ "$flag_all" -eq 1 ]; then
        [ "$flag_think" -eq 1 ] && think_set=("think") || { [ "$flag_nothink" -eq 1 ] && think_set=("nothink") || think_set=("nothink" "think"); }
        [ "$flag_curriculum" -eq 1 ] && curric_set=("curriculum") || { [ "$flag_standard" -eq 1 ] && curric_set=("standard") || curric_set=("standard" "curriculum"); }
    else
        local has_t=$((flag_think + flag_nothink)) has_c=$((flag_curriculum + flag_standard))
        if [ "$has_t" -eq 0 ] || [ "$has_c" -eq 0 ]; then
            echo "❌ Servono --think/--nothink + --curriculum/--standard, oppure --all."
            return 1
        fi
        [ "$flag_think" -eq 1 ]      && think_set=("think")
        [ "$flag_nothink" -eq 1 ]    && think_set=("nothink")
        [ "$flag_curriculum" -eq 1 ] && curric_set=("curriculum")
        [ "$flag_standard" -eq 1 ]   && curric_set=("standard")
    fi
    local base="$PROJ_DIR/experiments/checkpoints/grpo"
    for v in "${think_set[@]}"; do
        for c in "${curric_set[@]}"; do
            [ -d "$base/$v/$c" ] || continue
            echo "──── $v / $c ────"
            for model_dir in "$base/$v/$c"/*/; do
                [ -d "$model_dir" ] || continue
                local model=$(basename "$model_dir")
                echo "=== $model ==="
                # Checkpoint intermedi (per stage)
                for d in "$model_dir"stage_*/; do
                    if [ -d "$d" ]; then
                        echo "  $(basename "$d"):"
                        ls -d "$d"checkpoint-* 2>/dev/null | while read -r c2; do echo "    $(basename "$c2")"; done
                    fi
                done
                # Modelli finali (stages/)
                if [ -d "$model_dir/stages" ]; then
                    echo "  stages/:"
                    ls -d "$model_dir"stages/stage_*/ 2>/dev/null | while read -r s; do
                        echo "    $(basename "$s")"
                    done
                fi
                echo ""
            done
        done
    done
}

# Mostra tabella training log
# Uso: trainlog-table --think --curriculum [--tail N]
#       trainlog-table --nothink --all
#       trainlog-table --all
trainlog-table() {
    local flag_think=0 flag_nothink=0 flag_curriculum=0 flag_standard=0 flag_all=0
    local extra_args=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --think)      flag_think=1; shift ;;
            --nothink)    flag_nothink=1; shift ;;
            --curriculum) flag_curriculum=1; shift ;;
            --standard)   flag_standard=1; shift ;;
            --all)        flag_all=1; shift ;;
            *)            extra_args+=("$1"); shift ;;
        esac
    done
    if [ "$flag_think" -eq 1 ] && [ "$flag_nothink" -eq 1 ]; then
        echo "❌ --think e --nothink sono mutualmente esclusivi."; return 1
    fi
    if [ "$flag_curriculum" -eq 1 ] && [ "$flag_standard" -eq 1 ]; then
        echo "❌ --curriculum e --standard sono mutualmente esclusivi."; return 1
    fi
    local think_set=() curric_set=()
    if [ "$flag_all" -eq 1 ]; then
        [ "$flag_think" -eq 1 ] && think_set=("think") || { [ "$flag_nothink" -eq 1 ] && think_set=("nothink") || think_set=("nothink" "think"); }
        [ "$flag_curriculum" -eq 1 ] && curric_set=("curriculum") || { [ "$flag_standard" -eq 1 ] && curric_set=("standard") || curric_set=("standard" "curriculum"); }
    else
        local has_t=$((flag_think + flag_nothink)) has_c=$((flag_curriculum + flag_standard))
        if [ "$has_t" -eq 0 ] || [ "$has_c" -eq 0 ]; then
            echo "❌ Servono --think/--nothink + --curriculum/--standard, oppure --all."
            return 1
        fi
        [ "$flag_think" -eq 1 ]      && think_set=("think")
        [ "$flag_nothink" -eq 1 ]    && think_set=("nothink")
        [ "$flag_curriculum" -eq 1 ] && curric_set=("curriculum")
        [ "$flag_standard" -eq 1 ]   && curric_set=("standard")
    fi
    for v in "${think_set[@]}"; do
        for c in "${curric_set[@]}"; do
            cd "$PROJ_DIR" && python3 -m src.utils.show_training_log "experiments/checkpoints/grpo/$v/$c" "${extra_args[@]}"
        done
    done
}

# Genera grafici training con regressione polinomiale
# Uso: trainlog-plot --think --curriculum [--deg N]
#       trainlog-plot --all
trainlog-plot() {
    local flag_think=0 flag_nothink=0 flag_curriculum=0 flag_standard=0 flag_all=0
    local extra_args=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --think)      flag_think=1; shift ;;
            --nothink)    flag_nothink=1; shift ;;
            --curriculum) flag_curriculum=1; shift ;;
            --standard)   flag_standard=1; shift ;;
            --all)        flag_all=1; shift ;;
            *)            extra_args+=("$1"); shift ;;
        esac
    done
    if [ "$flag_think" -eq 1 ] && [ "$flag_nothink" -eq 1 ]; then
        echo "❌ --think e --nothink sono mutualmente esclusivi."; return 1
    fi
    if [ "$flag_curriculum" -eq 1 ] && [ "$flag_standard" -eq 1 ]; then
        echo "❌ --curriculum e --standard sono mutualmente esclusivi."; return 1
    fi
    local think_set=() curric_set=()
    if [ "$flag_all" -eq 1 ]; then
        [ "$flag_think" -eq 1 ] && think_set=("think") || { [ "$flag_nothink" -eq 1 ] && think_set=("nothink") || think_set=("nothink" "think"); }
        [ "$flag_curriculum" -eq 1 ] && curric_set=("curriculum") || { [ "$flag_standard" -eq 1 ] && curric_set=("standard") || curric_set=("standard" "curriculum"); }
    else
        local has_t=$((flag_think + flag_nothink)) has_c=$((flag_curriculum + flag_standard))
        if [ "$has_t" -eq 0 ] || [ "$has_c" -eq 0 ]; then
            echo "❌ Servono --think/--nothink + --curriculum/--standard, oppure --all."
            return 1
        fi
        [ "$flag_think" -eq 1 ]      && think_set=("think")
        [ "$flag_nothink" -eq 1 ]    && think_set=("nothink")
        [ "$flag_curriculum" -eq 1 ] && curric_set=("curriculum")
        [ "$flag_standard" -eq 1 ]   && curric_set=("standard")
    fi
    for v in "${think_set[@]}"; do
        for c in "${curric_set[@]}"; do
            cd "$PROJ_DIR" && python3 -m src.utils.show_training_log "experiments/checkpoints/grpo/$v/$c" --plot "${extra_args[@]}"
        done
    done
}

# Segui training live come tabella (uso: trainlog-live <JOB_ID>)
trainlog-live() {
    if [ -z "$1" ]; then
        echo "Uso: trainlog-live <JOB_ID>"
        return 1
    fi
    local logfile="$PROJ_DIR/logs/slurm-train-${1}.log"
    if [ ! -f "$logfile" ]; then
        echo "Log non trovato: $logfile"
        return 1
    fi
    tail -f "$logfile" | (cd "$PROJ_DIR" && python3 -u -m src.utils.live_training_table)
}

# Lancia training (uso: train --config PATH [extra args...])
train() {
    local config=""
    local extra_args=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            *) extra_args="$extra_args $1"; shift ;;
        esac
    done
    if [ -z "$config" ]; then
        echo "Uso: train --config PATH [extra args...]"
        echo ""
        echo "Config disponibili:"
        find "$PROJ_DIR"/experiments/configs -name 'grpo_*.yaml' -type f 2>/dev/null | sed "s|$PROJ_DIR/||" | sort | sed 's/^/  /'
        return 1
    fi
    cd "$PROJ_DIR" && CONFIG="$config" EXTRA_ARGS="$extra_args" sbatch cluster/train.sh
}

# Lancia eval manuale di un singolo modello.
# Il config YAML contiene il modello e i path dei checkpoint — l'eval script
# trova automaticamente l'ultimo checkpoint. --checkpoint sovrascrive.
#
# Esempi:
#   run-eval --config grpo_smollm2_135m.yaml                  # eval GRPO ultimo checkpoint
#   run-eval --config grpo_tinyllama.yaml --curriculum         # eval tutti gli stage + baseline
#   run-eval --config grpo_qwen05.yaml --compare               # eval + confronto con baseline
#   run-eval --config baseline.yaml                            # solo baseline (no checkpoint)
#   run-eval --config grpo_gemma2.yaml --checkpoint path/ckpt  # checkpoint specifico
run-eval() {
    local config=""
    local compare=0
    local curriculum=0
    local checkpoint=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --config) config="$2"; shift 2 ;;
            --compare) compare=1; shift ;;
            --curriculum) curriculum=1; shift ;;
            --checkpoint) checkpoint="$2"; shift 2 ;;
            --help|-h)
                echo "Uso: run-eval --config CONFIG [--compare] [--curriculum] [--checkpoint PATH]"
                echo ""
                echo "Opzioni:"
                echo "  --config CONFIG     Nome o path del config YAML (obbligatorio)"
                echo "  --compare           Confronta risultati GRPO vs baseline"
                echo "  --curriculum        Valuta tutti gli stage del curriculum (implica --compare)"
                echo "  --checkpoint PATH   Usa un checkpoint specifico (default: auto-detect ultimo)"
                echo ""
                echo "Config disponibili (path relativo al progetto):"
                find "$PROJ_DIR"/experiments/configs -name 'grpo_*.yaml' -o -name 'baseline.yaml' 2>/dev/null | sed "s|$PROJ_DIR/||" | sort | sed 's/^/  /'
                return 0
                ;;
            *) echo "❌ Argomento sconosciuto: $1"; echo "Usa: run-eval --help"; return 1 ;;
        esac
    done
    if [ -z "$config" ]; then
        echo "❌ --config mancante."
        echo ""
        echo "Uso: run-eval --config CONFIG [--compare] [--curriculum] [--checkpoint PATH]"
        echo ""
        echo "Config disponibili (path relativo al progetto):"
        find "$PROJ_DIR"/experiments/configs -name 'grpo_*.yaml' -o -name 'baseline.yaml' 2>/dev/null | sed "s|$PROJ_DIR/||" | sort | sed 's/^/  /'
        return 1
    fi
    # Se è un path relativo con sottocartelle, usalo direttamente
    if [[ "$config" != */* ]]; then
        # Cerca il file in tutte le sottocartelle di experiments/configs
        local found
        found=$(find "$PROJ_DIR/experiments/configs" -name "$config" -type f 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            config=$(echo "$found" | sed "s|$PROJ_DIR/||")
        else
            echo "❌ Config non trovato: $config"
            echo ""
            echo "Config disponibili:"
            find "$PROJ_DIR"/experiments/configs -name 'grpo_*.yaml' -o -name 'baseline.yaml' 2>/dev/null | sed "s|$PROJ_DIR/||" | sort | sed 's/^/  /'
            return 1
        fi
    fi
    if [ ! -f "$PROJ_DIR/$config" ]; then
        echo "❌ Config non trovato: $config"
        echo ""
        echo "Config disponibili:"
        find "$PROJ_DIR"/experiments/configs -name 'grpo_*.yaml' -o -name 'baseline.yaml' 2>/dev/null | sed "s|$PROJ_DIR/||" | sort | sed 's/^/  /'
        return 1
    fi
    cd "$PROJ_DIR" && CONFIG="$config" COMPARE="$compare" CURRICULUM="$curriculum" CHECKPOINT="$checkpoint" sbatch cluster/eval.sh
}

# Lancia tutti i modelli (train + eval)
# Uso: run-all [--think] [--standard] [--eval-only] [--train-only] [--models=1t,2e,3] [--resume]
run-all() {
    cd "$PROJ_DIR" && bash cluster/run_all.sh "$@"
}

# Controlla se il watcher è attivo
watcher-status() {
    # Controlla se c'è un fallimento
    if [ -f "$PROJ_DIR/.chain_failed" ]; then
        local failed=$(cat "$PROJ_DIR/.chain_failed")
        echo "❌ Pipeline FALLITA — job: $failed"
        echo "   Per riprendere: run-all --resume"
        return 1
    fi

    if [ -f "$PROJ_DIR/.chain_pid" ]; then
        local pid=$(cat "$PROJ_DIR/.chain_pid")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "✅ Watcher attivo (PID $pid)"
        else
            echo "❌ Watcher morto (PID $pid non trovato)"
            rm -f "$PROJ_DIR/.chain_pid"
            return 1
        fi
    else
        echo "❌ Nessun watcher attivo"
        return 1
    fi
}

# Uccidi il watcher
watcher-kill() {
    if [ -f "$PROJ_DIR/.chain_pid" ]; then
        local pid=$(cat "$PROJ_DIR/.chain_pid")
        if ! ps -p "$pid" > /dev/null 2>&1; then
            echo "⚠️  Watcher (PID $pid) già morto"
            rm -f "$PROJ_DIR/.chain_pid"
            return 0
        fi
        read -p "Uccidere il watcher (PID $pid)? [y/N] " confirm
        case "$confirm" in
            [yY]|[yY][eE][sS])
                if kill "$pid" 2>/dev/null; then
                    echo "✅ Watcher (PID $pid) terminato"
                else
                    echo "⚠️  Watcher (PID $pid) già morto"
                fi
                rm -f "$PROJ_DIR/.chain_pid"
                ;;
            *)
                echo "Annullato."
                ;;
        esac
    else
        echo "Nessun watcher attivo"
    fi
}

# Pulizia workspace (uso: clean [--force])
clean() {
    cd "$PROJ_DIR" && bash cluster/clean.sh "$@"
}

# Pulizia selettiva di un modello (uso: clean-model <TAG> [--all])
clean-model() {
    cd "$PROJ_DIR" && bash cluster/clean_model.sh "$@"
}

# Aggiungi job alla pipeline attiva (uso: chain-add [--think] [--models=1,3] [--eval-only] [--train-only])
chain-add() {
    cd "$PROJ_DIR" && bash cluster/run_all.sh --append "$@"
}

# Rimuovi job dalla pipeline attiva (uso: chain-remove --models=1,3)
chain-remove() {
    cd "$PROJ_DIR" && bash cluster/run_all.sh --remove "$@"
}

# Ferma la pipeline senza perdere lo stato (uso: chain-stop [--force])
# --force: uccidi watcher + cancella tutti i file di stato (ripartenza da zero)
chain-stop() {
    local force=0
    for arg in "$@"; do
        case "$arg" in
            --force) force=1 ;;
            --help|-h)
                echo "Uso: chain-stop [--force]"
                echo ""
                echo "  (default)   Ferma pipeline. Cancella job SLURM attivo + watcher."
                echo "              Salva lo stato per poter fare chain-start."
                echo "  --force     Come sopra, ma cancella TUTTI i file di stato."
                echo "              La pipeline dovrà essere ricominciata da capo."
                return 0
                ;;
        esac
    done

    cd "$PROJ_DIR"

    # 1. Trova il job SLURM attivo e le sue info
    local active_job=""
    local active_name=""
    active_job=$(squeue --me --noheader --format="%i %j" 2>/dev/null | head -1 | awk '{print $1}')
    active_name=$(squeue --me --noheader --format="%i %j" 2>/dev/null | head -1 | awk '{print $2}')

    # 2. Cancella il job SLURM attivo
    if [ -n "$active_job" ]; then
        scancel "$active_job" 2>/dev/null
        echo "✅ Job SLURM $active_job ($active_name) cancellato"
    else
        echo "⚠️  Nessun job SLURM attivo"
    fi

    # 3. Uccidi il watcher
    if [ -f .chain_pid ]; then
        local pid
        pid=$(cat .chain_pid)
        kill "$pid" 2>/dev/null && echo "✅ Watcher (PID $pid) terminato"
        rm -f .chain_pid
    fi

    if [ "$force" -eq 1 ]; then
        # --force: cancella tutto
        rm -f .job_chain .chain_pid .chain_failed .chain_stopped .monitor_cache
        echo "🗑️  File di stato cancellati (.job_chain, .chain_failed, .chain_stopped, .monitor_cache)"
        echo ""
        echo "Pipeline terminata definitivamente. Per ricominciare: run-all"
    else
        # Salva lo stato per chain-start
        # Determina il job che era attivo quando abbiamo fermato
        if [ -n "$active_name" ]; then
            local stopped_type=$(echo "$active_name" | cut -d- -f1)
            local stopped_tag=$(echo "$active_name" | cut -d- -f2-)

            # Cerca il config corrispondente
            local stopped_cfg=""

            # 1. Parse dal watcher log: "[chain] Sottometto: train gemma2-2b (experiments/configs/grpo_gemma2.yaml)"
            stopped_cfg=$(grep "Sottometto: ${stopped_type} ${stopped_tag} " logs/chain_watcher.log 2>/dev/null | tail -1 | sed -n 's/.*(\([^)]*\)).*/\1/p' || true)

            # 2. Fallback: derive from tag name (deterministic mapping)
            #    Tag format: base[-think][-cur] → path: {think|nothink}/{curriculum|standard}/grpo_*.yaml
            if [ -z "$stopped_cfg" ]; then
                local base_tag="$stopped_tag"
                local think_dir="nothink"
                local curric_dir="standard"
                # Strip suffixes to get base model name
                base_tag="${base_tag%-cur}"
                [[ "$stopped_tag" == *-cur* ]] && curric_dir="curriculum"
                base_tag="${base_tag%-think}"
                [[ "$stopped_tag" == *-think* ]] && think_dir="think"
                case "$base_tag" in
                    smollm2-135m) stopped_cfg="experiments/configs/${think_dir}/${curric_dir}/grpo_smollm2_135m.yaml" ;;
                    smollm2-360m) stopped_cfg="experiments/configs/${think_dir}/${curric_dir}/grpo_smollm2_360m.yaml" ;;
                    qwen25-05b)   stopped_cfg="experiments/configs/${think_dir}/${curric_dir}/grpo_qwen05.yaml" ;;
                    tinyllama-11b) stopped_cfg="experiments/configs/${think_dir}/${curric_dir}/grpo_tinyllama.yaml" ;;
                    gemma2-2b)    stopped_cfg="experiments/configs/${think_dir}/${curric_dir}/grpo_gemma2.yaml" ;;
                esac
            fi

            # Determine current stage for eval resume
            # Count only "Stage N: ... Pass@1:" lines (not Baseline, not comparison summary)
            local stopped_stage=0
            if [ "$stopped_type" = "eval" ] && [ -n "$active_job" ]; then
                local logfile="logs/slurm-eval-${active_job}.log"
                if [ -f "$logfile" ]; then
                    stopped_stage=$(grep -cP '^Stage \d+.*Pass@1:' "$logfile" 2>/dev/null || echo 0)
                fi
            fi

            echo "${stopped_type}:${stopped_cfg}:${stopped_tag}:${stopped_stage}:${active_job}" > .chain_stopped
            echo "💾 Stato salvato in .chain_stopped:"
            echo "   Tipo:  $stopped_type"
            echo "   Tag:   $stopped_tag"
            echo "   Config: ${stopped_cfg:-sconosciuto}"
            [ "$stopped_stage" -gt 0 ] && echo "   Stage completati: $stopped_stage"
        else
            echo "⚠️  Nessun job attivo da salvare"
            # Salva un marker vuoto per indicare che la pipeline è stata stoppata
            echo "none:::0:" > .chain_stopped
        fi
        rm -f .chain_failed
        echo ""
        echo "Pipeline fermata. Per riprendere: chain-start"
    fi
}

# Riprendi la pipeline dopo chain-stop (uso: chain-start)
chain-start() {
    cd "$PROJ_DIR"

    if [ ! -f .chain_stopped ]; then
        echo "❌ Nessun .chain_stopped trovato."
        echo "   chain-start funziona solo dopo chain-stop (senza --force)."
        echo "   Per una nuova pipeline: run-all"
        return 1
    fi

    local stopped_info
    stopped_info=$(cat .chain_stopped)
    local stopped_type=$(echo "$stopped_info" | cut -d: -f1)
    local stopped_cfg=$(echo "$stopped_info" | cut -d: -f2)
    local stopped_tag=$(echo "$stopped_info" | cut -d: -f3)
    local stopped_stage=$(echo "$stopped_info" | cut -d: -f4)
    local stopped_slurm=$(echo "$stopped_info" | cut -d: -f5)

    # Se il config è vuoto, prova a derivarlo dal tag
    if [ -z "$stopped_cfg" ] && [ "$stopped_type" != "none" ]; then
        local base_tag="$stopped_tag"
        local think_dir="nothink"
        local curric_dir="standard"
        base_tag="${base_tag%-cur}"
        [[ "$stopped_tag" == *-cur* ]] && curric_dir="curriculum"
        base_tag="${base_tag%-think}"
        [[ "$stopped_tag" == *-think* ]] && think_dir="think"
        case "$base_tag" in
            smollm2-135m) stopped_cfg="experiments/configs/${think_dir}/${curric_dir}/grpo_smollm2_135m.yaml" ;;
            smollm2-360m) stopped_cfg="experiments/configs/${think_dir}/${curric_dir}/grpo_smollm2_360m.yaml" ;;
            qwen25-05b)   stopped_cfg="experiments/configs/${think_dir}/${curric_dir}/grpo_qwen05.yaml" ;;
            tinyllama-11b) stopped_cfg="experiments/configs/${think_dir}/${curric_dir}/grpo_tinyllama.yaml" ;;
            gemma2-2b)    stopped_cfg="experiments/configs/${think_dir}/${curric_dir}/grpo_gemma2.yaml" ;;
        esac
    fi

    # Guard: non procedere senza config
    if [ -z "$stopped_cfg" ] && [ "$stopped_type" != "none" ]; then
        echo "❌ Config non trovato per $stopped_type $stopped_tag."
        echo "   File .chain_stopped corrotto. Cancella e riparti: rm .chain_stopped && run-all"
        return 1
    fi

    echo "============================================"
    echo "  CHAIN START — Ripresa pipeline"
    echo "  Date:      $(date)"
    echo "  Stopped:   $stopped_type $stopped_tag"
    echo "  Config:    ${stopped_cfg:-sconosciuto}"
    [ "$stopped_stage" -gt 0 ] && echo "  Stage completati: $stopped_stage"
    echo "============================================"
    echo ""

    if [ "$stopped_type" = "none" ]; then
        echo "ℹ️  La pipeline era già in pausa (nessun job attivo al momento dello stop)."
        echo "   Riavvio il watcher con i job rimanenti in .job_chain."
    elif [ "$stopped_type" = "train" ]; then
        # Reinserisci il train con --resume + eval in testa alla catena
        local RESUME_CHAIN
        RESUME_CHAIN=$(mktemp)
        echo "train:${stopped_cfg}:${stopped_tag}:--resume" > "$RESUME_CHAIN"
        # Aggiungi eval se non è già il prossimo nella catena
        local next_in_chain=""
        [ -f .job_chain ] && [ -s .job_chain ] && next_in_chain=$(head -1 .job_chain)
        local next_type=$(echo "$next_in_chain" | cut -d: -f1)
        local next_tag=$(echo "$next_in_chain" | cut -d: -f3)
        if [ "$next_type" != "eval" ] || [ "$next_tag" != "$stopped_tag" ]; then
            echo "eval:${stopped_cfg}:${stopped_tag}" >> "$RESUME_CHAIN"
        fi
        [ -f .job_chain ] && [ -s .job_chain ] && cat .job_chain >> "$RESUME_CHAIN"
        mv "$RESUME_CHAIN" .job_chain
        echo "→ Training $stopped_tag verrà ripreso dall'ultimo checkpoint"
        echo "→ Eval $stopped_tag verrà eseguito dopo il train"
    elif [ "$stopped_type" = "eval" ]; then
        # Reinserisci l'eval con --skip-stages in testa alla catena
        local RESUME_CHAIN
        RESUME_CHAIN=$(mktemp)
        if [ "$stopped_stage" -gt 0 ]; then
            echo "eval:${stopped_cfg}:${stopped_tag}:--skip-stages=${stopped_stage}" > "$RESUME_CHAIN"
            echo "→ Eval $stopped_tag riprende da stage $((stopped_stage + 1)) (skip $stopped_stage completati)"
        else
            echo "eval:${stopped_cfg}:${stopped_tag}" > "$RESUME_CHAIN"
            echo "→ Eval $stopped_tag verrà rieseguito da capo"
        fi
        [ -f .job_chain ] && [ -s .job_chain ] && cat .job_chain >> "$RESUME_CHAIN"
        mv "$RESUME_CHAIN" .job_chain
    fi

    rm -f .chain_stopped .chain_failed

    if [ ! -f .job_chain ] || [ ! -s .job_chain ]; then
        echo ""
        echo "⚠️  Nessun job rimanente nella catena."
        echo "   Per una nuova pipeline: run-all"
        return 0
    fi

    local TOTAL
    TOTAL=$(wc -l < .job_chain)
    echo ""
    echo "Catena ($TOTAL job):"
    cat -n .job_chain
    echo ""

    # Uccidi eventuale watcher residuo
    if [ -f .chain_pid ]; then
        local OLD_PID
        OLD_PID=$(cat .chain_pid)
        kill "$OLD_PID" 2>/dev/null
        rm -f .chain_pid
    fi

    mkdir -p logs
    nohup bash cluster/chain_next.sh >> logs/chain_watcher.log 2>&1 &
    local WATCHER_PID=$!
    echo "$WATCHER_PID" > .chain_pid

    echo "============================================"
    echo "  Pipeline ripresa!"
    echo "  Watcher PID: $WATCHER_PID"
    echo "  Log: logs/chain_watcher.log"
    echo "============================================"
}

# Mostra la catena di job attuale (uso: chain-show)
chain-show() {
    watcher-status
    echo ""
    if [ -f "$PROJ_DIR/.chain_stopped" ]; then
        local info
        info=$(cat "$PROJ_DIR/.chain_stopped")
        local st_type=$(echo "$info" | cut -d: -f1)
        local st_tag=$(echo "$info" | cut -d: -f3)
        [ "$st_type" != "none" ] && echo "⏸️  Pipeline fermata su: $st_type $st_tag"
        echo "   Per riprendere: chain-start"
        echo ""
    fi
    if [ ! -f "$PROJ_DIR/.job_chain" ] || [ ! -s "$PROJ_DIR/.job_chain" ]; then
        echo "Nessun job in coda."
        return 0
    fi
    local total
    total=$(wc -l < "$PROJ_DIR/.job_chain")
    echo "Job in coda ($total):"
    cat -n "$PROJ_DIR/.job_chain"
}

# Monitor live della pipeline (uso: monitor [--poll N])
monitor() {
    cd "$PROJ_DIR" && python3 -u -m src.utils.chain_monitor "$@"
}

# ── Pip / Environment ────────────────────────────────────────────────────────

# Pulisci tutti i pacchetti --user
pip-clean() {
    echo "🗑️  Rimozione pacchetti pip --user..."
    rm -rf ~/.local/lib/python3.*/site-packages/*
    rm -rf ~/.local/bin/*
    echo "✅ ~/.local ripulito"
}

# (Re)installa dipendenze da setup.sh
pip-setup() {
    echo "📦 Installazione dipendenze..."
    cd "$PROJ_DIR" && bash cluster/setup.sh
}

# Pulisci e reinstalla da zero
pip-reset() {
    pip-clean
    pip-setup
}

# ── Meta ─────────────────────────────────────────────────────────────────────

# Lista di tutti i comandi custom registrati
_GRPO_ALIASES="myjobs jobinfo killjob killalljobs trainlog evallog baselog lastlog tree ltree gpu quota proj ckpts trainlog-table trainlog-plot trainlog-live train run-eval run-all watcher-status watcher-kill clean clean-model chain-add chain-remove chain-stop chain-start chain-show monitor pip-clean pip-setup pip-reset unload-aliases install-aliases uninstall-aliases"

# Mostra i comandi disponibili
claudio() {
    echo "Comandi GRPO disponibili:"
    echo ""
    echo "── Job management ──"
    echo "   myjobs            — lista job attivi"
    echo "   jobinfo <ID>      — dettagli job"
    echo "   killjob <ID>      — cancella job"
    echo "   killalljobs       — cancella tutti i miei job + watcher"
    echo ""
    echo "── Log monitoring ──"
    echo "   trainlog <ID>     — segui log training"
    echo "   evallog <ID>      — segui log eval"
    echo "   baselog <ID>      — segui log baseline"
    echo "   lastlog [N]       — segui l'ultimo log (N=ultime N righe)"
    echo ""
    echo "── Training & eval ──"
    echo "   train --config PATH [extra args...]"
    echo "                     — lancia training singolo"
    echo "   run-eval --config PATH [--compare] [--curriculum] [--checkpoint PATH]"
    echo "                     — lancia evaluation singola"
    echo "   run-all <VARIANTE> [--eval-only] [--train-only] [--models=SPEC] [--resume]"
    echo "                     — lancia pipeline multi-modello"
    echo "                       VARIANTE: --think/--nothink + --curriculum/--standard, o --all"
    echo ""
    echo "── Pipeline (chain) ──"
    echo "   chain-show        — mostra stato pipeline + job in coda"
    echo "   chain-add <VARIANTE> [--models=1,3] [--eval-only]"
    echo "                     — aggiungi job alla pipeline attiva"
    echo "   chain-remove --models=1,3"
    echo "                     — rimuovi job dalla coda"
    echo "   chain-stop        — ferma pipeline (preserva stato per resume)"
    echo "   chain-stop --force"
    echo "                     — ferma + cancella tutti i file di stato"
    echo "   chain-start       — riprendi pipeline dopo chain-stop"
    echo "   watcher-status    — controlla se il watcher è attivo"
    echo "   watcher-kill      — uccidi il watcher (senza salvare stato)"
    echo ""
    echo "── Monitor ──"
    echo "   monitor [--poll N] [--tab] [--samples [N]] [--metrics] [--all [N]]"
    echo "                     — monitor live della pipeline"
    echo "                       --all [N] = --tab --metrics --samples [N]"
    echo ""
    echo "── Analisi ──"
    echo "   ckpts <VARIANTE>  — mostra checkpoint"
    echo "                       VARIANTE: --think/--nothink + --curriculum/--standard, o --all"
    echo "   trainlog-table <VARIANTE> [--tail N]"
    echo "                     — tabella metriche training"
    echo "   trainlog-plot <VARIANTE> [--deg N]"
    echo "                     — grafici training con regressione polinomiale"
    echo "   trainlog-live <ID> — training live come tabella"
    echo ""
    echo "── Utilità ──"
    echo "   proj              — cd al progetto"
    echo "   tree <DIR> [N]    — albero cartelle (profondità N)"
    echo "   ltree <DIR>       — albero cartelle compatto"
    echo "   gpu               — stato GPU"
    echo "   quota             — uso disco progetto"
    echo "   clean             — pulizia workspace (dry-run, usa --force per cancellare)"
    echo "   clean-model <TAG> <VARIANTE> [--grpo|--baseline|--sft|--data-all]"
    echo "                     — pulisci checkpoints/logs di un modello"
    echo ""
    echo "── Pip / Environment ──"
    echo "   pip-clean         — rimuovi tutti i pacchetti pip --user"
    echo "   pip-setup         — (re)installa dipendenze (cluster/setup.sh)"
    echo "   pip-reset         — pip-clean + pip-setup"
    echo ""
    echo "── Meta ──"
    echo "   claudio           — mostra questo messaggio"
    echo "   unload-aliases    — rimuovi alias (sessione corrente)"
    echo "   install-aliases   — aggiungi alias al .bashrc (permanente)"
    echo "   uninstall-aliases — rimuovi alias dal .bashrc (permanente)"
}

# Rimuovi tutti gli alias e funzioni custom (solo sessione corrente)
unload-aliases() {
    for cmd in $_GRPO_ALIASES; do
        unalias "$cmd" 2>/dev/null
        unset -f "$cmd" 2>/dev/null
    done
    unset _GRPO_ALIASES PROJ_DIR
    echo "✅ Alias GRPO rimossi (sessione corrente)."
}

_ALIASES_SOURCE_LINE="source ~/GRPO-strict-generation/cluster/aliases.sh"

# Aggiungi alias al .bashrc (caricati ad ogni login)
install-aliases() {
    if grep -qF "$_ALIASES_SOURCE_LINE" ~/.bashrc 2>/dev/null; then
        echo "⚠️  Alias già presenti in ~/.bashrc"
    else
        echo "$_ALIASES_SOURCE_LINE" >> ~/.bashrc
        echo "✅ Alias aggiunti a ~/.bashrc (attivi dal prossimo login)"
    fi
}

# Rimuovi alias dal .bashrc (non più caricati al login)
uninstall-aliases() {
    if grep -qF "$_ALIASES_SOURCE_LINE" ~/.bashrc 2>/dev/null; then
        sed -i "\|$_ALIASES_SOURCE_LINE|d" ~/.bashrc
        echo "✅ Alias rimossi da ~/.bashrc"
    else
        echo "⚠️  Alias non presenti in ~/.bashrc"
    fi
    unload-aliases
}

echo "✅ Alias GRPO caricati. Digita 'claudio' per la lista comandi."
