#!/bin/bash
# ============================================================================
# Pulizia selettiva — rimuove checkpoints e logs di un modello specifico.
#
# Uso:
#   bash cluster/clean_model.sh                                          # dry-run (mostra cosa c'è)
#   bash cluster/clean_model.sh <MODEL_TAG> --think --curriculum          # cancella GRPO think/curriculum
#   bash cluster/clean_model.sh <MODEL_TAG> --nothink --standard          # cancella GRPO nothink/standard
#   bash cluster/clean_model.sh <MODEL_TAG> --think --all                 # cancella tutto think
#   bash cluster/clean_model.sh <MODEL_TAG> --all                         # cancella tutto
#
# MODEL_TAG è il nome usato nei path (es. smollm2-135m, tinyllama-11b, ...).
# Senza MODEL_TAG: mostra un riepilogo di cosa esiste per ogni modello (dry-run).
#
# Selezione variante (obbligatoria, come run_all.sh):
#   --think / --nothink          Modalità thinking
#   --curriculum / --standard    Tipo di training
#   --all                        Wildcard per dimensione mancante (o entrambe)
#
# Selezione dati:
#   --grpo       Solo dati GRPO (default se nessun flag specifico)
#   --baseline   Solo dati baseline
#   --sft        Solo dati SFT
#   --data-all   GRPO + baseline + SFT
#
# Esempi:
#   bash cluster/clean_model.sh                                           # dry-run
#   bash cluster/clean_model.sh tinyllama-11b --nothink --curriculum       # GRPO nothink/curriculum
#   bash cluster/clean_model.sh tinyllama-11b --all --data-all             # tutto di ogni variante
#   bash cluster/clean_model.sh gemma2-2b --think --all                   # GRPO think (curric+standard)
#   bash cluster/clean_model.sh smollm2-135m --nothink --curriculum --baseline  # solo baseline nothink/curriculum
# ============================================================================

set -e
cd "$HOME/GRPO-strict-generation"

# ── Modelli validi (per validazione) ──────────────────────────────────────────
VALID_MODELS=("smollm2-135m" "smollm2-360m" "qwen25-05b" "tinyllama-11b" "gemma2-2b")
THINK_VARIANTS=("nothink" "think")

# Mapping tag → HuggingFace short name (per baseline subfolder)
baseline_hf_name() {
    case "$1" in
        smollm2-135m)  echo "SmolLM2-135M-Instruct" ;;
        smollm2-360m)  echo "SmolLM2-360M-Instruct" ;;
        qwen25-05b)    echo "Qwen2.5-0.5B-Instruct" ;;
        tinyllama-11b) echo "TinyLlama-1.1B-Chat-v1.0" ;;
        gemma2-2b)     echo "gemma-2-2b-it" ;;
        *)             echo "" ;;  # modello custom — non sappiamo il nome HF
    esac
}

# ── Parsing argomenti ─────────────────────────────────────────────────────────
MODEL=""
DO_GRPO=0
DO_BASELINE=0
DO_SFT=0
FLAG_THINK=0
FLAG_NOTHINK=0
FLAG_CURRICULUM=0
FLAG_STANDARD=0
FLAG_ALL=0

for arg in "$@"; do
    case "$arg" in
        --grpo)       DO_GRPO=1 ;;
        --baseline)   DO_BASELINE=1 ;;
        --sft)        DO_SFT=1 ;;
        --data-all)   DO_GRPO=1; DO_BASELINE=1; DO_SFT=1 ;;
        --think)      FLAG_THINK=1 ;;
        --nothink)    FLAG_NOTHINK=1 ;;
        --curriculum) FLAG_CURRICULUM=1 ;;
        --standard)   FLAG_STANDARD=1 ;;
        --all)        FLAG_ALL=1 ;;
        --help|-h)
            sed -n '2,/^# ====.*===$/p' "$0" | head -n -1 | sed 's/^# \?//'
            echo ""
            echo "Modelli disponibili:"
            for m in "${VALID_MODELS[@]}"; do echo "  $m"; done
            exit 0
            ;;
        -*)
            echo "❌ Opzione sconosciuta: $arg"
            echo "Uso: bash cluster/clean_model.sh <MODEL_TAG> <VARIANTE> [--grpo|--baseline|--sft|--data-all]"
            exit 1
            ;;
        *)
            if [ -z "$MODEL" ]; then
                MODEL="$arg"
            else
                echo "❌ Troppi argomenti posizionali: $arg"
                exit 1
            fi
            ;;
    esac
done

# Default: solo GRPO se nessun flag specifico
if [ $DO_GRPO -eq 0 ] && [ $DO_BASELINE -eq 0 ] && [ $DO_SFT -eq 0 ]; then
    DO_GRPO=1
fi

# ── Validazione flag variante (come run_all.sh) ──────────────────────────────
if [ "$FLAG_THINK" -eq 1 ] && [ "$FLAG_NOTHINK" -eq 1 ]; then
    echo "❌ --think e --nothink sono mutualmente esclusivi."
    exit 1
fi
if [ "$FLAG_CURRICULUM" -eq 1 ] && [ "$FLAG_STANDARD" -eq 1 ]; then
    echo "❌ --curriculum e --standard sono mutualmente esclusivi."
    exit 1
fi

THINK_VARIANTS_FILTER=()
CURRIC_VARIANTS_FILTER=()

if [ "$FLAG_ALL" -eq 1 ]; then
    if [ "$FLAG_THINK" -eq 1 ]; then
        THINK_VARIANTS_FILTER=("think")
    elif [ "$FLAG_NOTHINK" -eq 1 ]; then
        THINK_VARIANTS_FILTER=("nothink")
    else
        THINK_VARIANTS_FILTER=("nothink" "think")
    fi
    if [ "$FLAG_CURRICULUM" -eq 1 ]; then
        CURRIC_VARIANTS_FILTER=("curriculum")
    elif [ "$FLAG_STANDARD" -eq 1 ]; then
        CURRIC_VARIANTS_FILTER=("standard")
    else
        CURRIC_VARIANTS_FILTER=("standard" "curriculum")
    fi
elif [ -n "$MODEL" ]; then
    # Varianti obbligatorie quando si cancella (non nel dry-run)
    HAS_THINK=$((FLAG_THINK + FLAG_NOTHINK))
    HAS_CURRIC=$((FLAG_CURRICULUM + FLAG_STANDARD))
    if [ "$HAS_THINK" -eq 0 ] || [ "$HAS_CURRIC" -eq 0 ]; then
        echo "❌ Servono 2 flag: uno tra --think/--nothink e uno tra --curriculum/--standard."
        echo "   Oppure usa --all per coprire la dimensione mancante."
        echo ""
        echo "   Esempi:"
        echo "     --think --curriculum       una combinazione specifica"
        echo "     --nothink --all            tutto nothink"
        echo "     --all                      tutto"
        exit 1
    fi
    [ "$FLAG_THINK" -eq 1 ]      && THINK_VARIANTS_FILTER=("think")
    [ "$FLAG_NOTHINK" -eq 1 ]    && THINK_VARIANTS_FILTER=("nothink")
    [ "$FLAG_CURRICULUM" -eq 1 ] && CURRIC_VARIANTS_FILTER=("curriculum")
    [ "$FLAG_STANDARD" -eq 1 ]   && CURRIC_VARIANTS_FILTER=("standard")
fi

# ── Dry-run (nessun modello specificato) ──────────────────────────────────────
if [ -z "$MODEL" ]; then
    echo "=== DRY RUN — mostra cosa esiste per ogni modello ==="
    echo ""
    for m in "${VALID_MODELS[@]}"; do
        DIRS_FOUND=()
        # GRPO (nothink + think) × (curriculum + standard)
        for variant in "${THINK_VARIANTS[@]}"; do
            for curric in curriculum standard; do
                [ -d "experiments/checkpoints/grpo/$variant/$curric/$m" ] && DIRS_FOUND+=("checkpoints/grpo/$variant/$curric/$m")
                [ -d "experiments/logs/grpo/$variant/$curric/$m" ] && DIRS_FOUND+=("logs/grpo/$variant/$curric/$m")
            done
        done
        # Baseline
        HF=$(baseline_hf_name "$m")
        [ -n "$HF" ] && [ -d "experiments/logs/baseline/$HF" ] && DIRS_FOUND+=("logs/baseline/$HF")
        # SFT
        [ -d "experiments/checkpoints/sft/$m" ] && DIRS_FOUND+=("checkpoints/sft/$m")
        [ -d "experiments/logs/sft/$m" ] && DIRS_FOUND+=("logs/sft/$m")

        if [ ${#DIRS_FOUND[@]} -gt 0 ]; then
            echo "  $m:"
            for d in "${DIRS_FOUND[@]}"; do
                SIZE=$(du -sh "experiments/$d" 2>/dev/null | cut -f1)
                echo "    experiments/$d ($SIZE)"
            done
        else
            echo "  $m: (niente)"
        fi
    done
    echo ""
    echo "Per cancellare: bash cluster/clean_model.sh <MODEL_TAG> <VARIANTE> [--grpo|--baseline|--sft|--data-all]"
    exit 0
fi

# ── Validazione modello ──────────────────────────────────────────────────────

FOUND=0
for m in "${VALID_MODELS[@]}"; do
    [ "$m" = "$MODEL" ] && FOUND=1
done
if [ $FOUND -eq 0 ]; then
    echo "⚠️  '$MODEL' non è un modello noto. Modelli validi:"
    for m in "${VALID_MODELS[@]}"; do echo "  $m"; done
    echo ""
    echo "Procedo comunque (potrebbe essere un modello custom)..."
fi

echo "Pulizia modello: $MODEL"
echo "Scope: $([ $DO_GRPO -eq 1 ] && echo 'GRPO ')$([ $DO_BASELINE -eq 1 ] && echo 'BASELINE ')$([ $DO_SFT -eq 1 ] && echo 'SFT ')[${THINK_VARIANTS_FILTER[*]}×${CURRIC_VARIANTS_FILTER[*]}]"
echo ""

CLEANED=0

# ── GRPO ──────────────────────────────────────────────────────────────────────
if [ $DO_GRPO -eq 1 ]; then
    for variant in "${THINK_VARIANTS_FILTER[@]}"; do
        for curric in "${CURRIC_VARIANTS_FILTER[@]}"; do
            # Checkpoints: experiments/checkpoints/grpo/<variant>/<curric>/<model>/
            DIR="experiments/checkpoints/grpo/$variant/$curric/$MODEL"
            if [ -d "$DIR" ]; then
                echo "[GRPO/$variant/$curric] Checkpoints: $DIR"
                rm -rf "$DIR"
                CLEANED=1
            else
                echo "[GRPO/$variant/$curric] Checkpoints: $DIR (non esiste — skip)"
            fi

            # Logs + eval + wandb: experiments/logs/grpo/<variant>/<curric>/<model>/
            DIR="experiments/logs/grpo/$variant/$curric/$MODEL"
            if [ -d "$DIR" ]; then
                echo "[GRPO/$variant/$curric] Logs/eval:   $DIR"
                rm -rf "$DIR"
                CLEANED=1
            else
                echo "[GRPO/$variant/$curric] Logs/eval:   $DIR (non esiste — skip)"
            fi
        done
    done
fi

# ── Baseline ──────────────────────────────────────────────────────────────────
if [ $DO_BASELINE -eq 1 ]; then
    # Baseline results: experiments/logs/baseline/<HF_short_name>/
    HF_NAME=$(baseline_hf_name "$MODEL")
    if [ -n "$HF_NAME" ]; then
        DIR="experiments/logs/baseline/$HF_NAME"
        if [ -d "$DIR" ]; then
            echo "[BASELINE] Eval: $DIR"
            rm -rf "$DIR"
            CLEANED=1
        else
            echo "[BASELINE] Eval: $DIR (non esiste — skip)"
        fi
    else
        echo "[BASELINE] ⚠️  Modello custom '$MODEL' — nome HF sconosciuto, skip baseline."
    fi
fi

# ── SFT ───────────────────────────────────────────────────────────────────────
if [ $DO_SFT -eq 1 ]; then
    # Checkpoints: experiments/checkpoints/sft/<model>/
    DIR="experiments/checkpoints/sft/$MODEL"
    if [ -d "$DIR" ]; then
        echo "[SFT] Checkpoints: $DIR"
        rm -rf "$DIR"
        CLEANED=1
    else
        echo "[SFT] Checkpoints: $DIR (non esiste — skip)"
    fi

    # Logs: experiments/logs/sft/<model>/
    DIR="experiments/logs/sft/$MODEL"
    if [ -d "$DIR" ]; then
        echo "[SFT] Logs: $DIR"
        rm -rf "$DIR"
        CLEANED=1
    else
        echo "[SFT] Logs: $DIR (non esiste — skip)"
    fi
fi

# ── Riepilogo ─────────────────────────────────────────────────────────────────
echo ""
if [ $CLEANED -eq 1 ]; then
    echo "✅ Pulizia completata per '$MODEL'."
else
    echo "ℹ️  Nessuna cartella da pulire per '$MODEL'."
fi
