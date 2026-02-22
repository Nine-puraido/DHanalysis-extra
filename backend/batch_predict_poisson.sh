#!/bin/bash
# Batch predict with Poisson model for world league fixtures missing Poisson predictions
# 586 fixtures in batches of 10

cd /Users/puraidointern/DHextra/backend
source .venv/bin/activate

# All fixture IDs needing Poisson predictions
ALL_IDS=(56 110 121 $(seq 3022 3604))

BATCH_SIZE=10
TOTAL=${#ALL_IDS[@]}
BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
OK=0
FAIL=0

echo "Starting Poisson batch predictions: $TOTAL fixtures in batches of $BATCH_SIZE"

for (( i=0; i<TOTAL; i+=BATCH_SIZE )); do
    BATCH_NUM=$(( i/BATCH_SIZE + 1 ))
    # Slice the array
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
