#!/bin/bash

# Need that dev mode to get the /server_info endpoint to get the maximum model length
model=openai/gpt-oss-20b
API_PORT=${API_PORT:-8002}

$RUN_PREFIX env VLLM_SERVER_DEV_MODE=1 uv run vllm serve \
    "$model" \
    --host 0.0.0.0 \
    --port $API_PORT \
    --max-model-len $((1 << 14)) \
    --max-num-seqs 16 \
    --max-num-batched-tokens $((1 << 13)) \
    --gpu-memory-utilization 0.85
