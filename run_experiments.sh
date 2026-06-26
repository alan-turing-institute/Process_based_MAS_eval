#!/bin/bash

SCRIPT_DIR="/home/c3047064/VSC Projects/ATI/hanabi-learning-environment"
BATCH_COOLDOWN=600  # seconds to wait between runs to avoid rate limits

cd "$SCRIPT_DIR"

model="Kimi-K2.5"
for i in $(seq 1 6); do
    echo "Starting run ${i}/6 for model: ${model}"
    python single_agent_hanabi.py --model "${model}" --context_level 1 --weighting True > "logs/stdout/Kimi_K2.5_run${i}.txt" 2>&1
    echo "Run ${i}/6 complete. Cooling down for ${BATCH_COOLDOWN}s..."
    sleep $BATCH_COOLDOWN
done

echo "All runs complete."
