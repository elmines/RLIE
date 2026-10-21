#!/bin/bash

set -x
ROUTER_PORT=${ROUTER_PORT:-8001}
WORKER_URLS=($@)
if [ -z "${WORKER_URLS[*]}" ]
then
    WORKER_URLS=(http://0.0.0.0:8002)
fi

$RUN_PREFIX uv run vllm-router \
    --host 0.0.0.0 \
    --port $ROUTER_PORT \
    --worker-urls "${WORKER_URLS[@]}" \
    --policy cache_aware