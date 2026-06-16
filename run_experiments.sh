#!/bin/bash

SCRIPT_DIR="/home/c3047064/VSC Projects/ATI/hanabi-learning-environment"
BATCH_COOLDOWN=600  # seconds to wait between batches to avoid rate limits
KILL_LOG="${SCRIPT_DIR}/logs/killed_sessions.log"

run_batch() {
    local model=$1
    local batch=$2
    # Replace special chars in model name for valid tmux session names
    local safe_model="${model//-/_}"
    safe_model="${safe_model//./_}"
    local sessions=()

    echo "Starting batch ${batch} for model: ${model}"

    for i in $(seq 1 3); do
        local session="${safe_model}_b${batch}_r${i}"
        if tmux has-session -t "$session" 2>/dev/null; then
            echo "  WARNING: stale session found, killing: $session"
            echo "$(date '+%Y-%m-%d %H:%M:%S') | killed stale session: $session" >> "$KILL_LOG"
            tmux kill-session -t "$session"
        fi
        tmux new-session -d -s "$session" -c "$SCRIPT_DIR" \
            "python single_agent_hanabi.py --model '${model}' > logs/stdout/${session}.txt 2>&1; exit"
        sessions+=("$session")
        echo "  Launched: $session"
    done

    echo "  Waiting for batch ${batch} to finish..."
    while true; do
        running=0
        for session in "${sessions[@]}"; do
            if tmux has-session -t "$session" 2>/dev/null; then
                ((running++))
            fi
        done
        [ $running -eq 0 ] && break
        echo "  $running session(s) still running..."
        sleep 300
    done

    echo "  Batch ${batch} for ${model} complete."
    echo "  Cooling down for ${BATCH_COOLDOWN}s..."
    sleep $BATCH_COOLDOWN
}

for model in "DeepSeek-R1"; do
    run_batch "$model" 1
    run_batch "$model" 2
    run_batch "$model" 3
    run_batch "$model" 4
    run_batch "$model" 5
    run_batch "$model" 6
    run_batch "$model" 7
done

echo "All runs complete."