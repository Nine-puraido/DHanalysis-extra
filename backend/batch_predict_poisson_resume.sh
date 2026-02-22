#!/bin/bash
# Resume Poisson batch predictions from fixture 3129 onwards
# (batches 1-10 completed = fixtures 56,110,121,3022-3128)

cd /Users/puraidointern/DHextra/backend
source .venv/bin/activate

# Remaining fixture IDs (3129-3604)
ALL_IDS=($(seq 3129 3604))

BATCH_SIZE=10
TOTAL=${#ALL_IDS[@]}
BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
OK=0
FAIL=0

echo "Resuming Poisson batch predictions: $TOTAL fixtures in batches of $BATCH_SIZE"

for (( i=0; i<TOTAL; i+=BATCH_SIZE )); do
    BATCH_NUM=$(( i/BATCH_SIZE + 1 ))
    CHUNK=("${ALL_IDS[@]:i:BATCH_SIZE}")
    IDS_STR=$(IFS=,; echo "${CHUNK[*]}")

    echo -n "[$BATCH_NUM/$BATCHES] Predicting fixtures: ${IDS_STR:0:60}..."

    if python -m dhx.modeling predict-upcoming --fixture-ids "$IDS_STR" > /dev/null 2>&1; then
        OK=$(( OK + ${#CHUNK[@]} ))
        echo "  OK ($OK/$TOTAL done)"
    else
        FAIL=$(( FAIL + ${#CHUNK[@]} ))
        echo "  FAILED (batch $BATCH_NUM)"
    fi

    sleep 2
done

echo ""
echo "=== COMPLETE ==="
echo "Predicted: $OK"
echo "Failed: $FAIL"
echo "Total: $TOTAL"
