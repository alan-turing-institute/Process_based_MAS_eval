#!/bin/bash

SCRIPT_DIR="/home/c3047064/hanabi-learning-environment"
BATCH_COOLDOWN=600  # seconds to wait between runs to avoid rate limits

cd "$SCRIPT_DIR"

model="Kimi-K2.5"

for i in $(seq 1 20); do
    echo "Starting run ${i}/20 for model: ${model} Weighting: True Weight Type: fullcoop"
    python single_agent_hanabi.py --model "${model}" --context_level 1 --weighting True --weight_type fullcoop > "logs/stdout/Kimi_K2.5_fullcoop_run${i}.txt" 2>&1
    echo "Run ${i}/20 complete for model: ${model} Weighting: True Weight Type: fullcoop. Cooling down for ${BATCH_COOLDOWN}s..."
    sleep $BATCH_COOLDOWN
done

for i in $(seq 1 20); do
    echo "Starting run ${i}/20 for model: ${model} Weighting: True Weight Type: mixcoop"
    python single_agent_hanabi.py --model "${model}" --context_level 1 --weighting True --weight_type mixcoop > "logs/stdout/Kimi_K2.5_mixcoop_run${i}.txt" 2>&1
    echo "Run ${i}/20 complete for model: ${model} Weighting: True Weight Type: mixcoop. Cooling down for ${BATCH_COOLDOWN}s..."
    sleep $BATCH_COOLDOWN
done

for i in $(seq 1 20); do
    echo "Starting run ${i}/20 for model: ${model} Weighting: True Weight Type: fullcomp"
    python single_agent_hanabi.py --model "${model}" --context_level 1 --weighting True --weight_type fullcomp > "logs/stdout/Kimi_K2.5_fullcomp_run${i}.txt" 2>&1
    echo "Run ${i}/20 complete for model: ${model} Weighting: True Weight Type: fullcomp. Cooling down for ${BATCH_COOLDOWN}s..."
    sleep $BATCH_COOLDOWN
done

echo "All runs complete."
