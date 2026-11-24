#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUT_ROOT="${OUT_ROOT:-runs}"
FAULT_MODE="${FAULT_MODE:-clean}"
FAULT_P="${FAULT_P:-0.2}"
FAULT_SEED="${FAULT_SEED:-0}"
TP="${TP:-8}"

# CRITICAL: export so launcher AND runner see these
export FAULT_MODE FAULT_P FAULT_SEED

# Shared state between launcher and tests — MUST be set before server starts
mkdir -p "$OUT_ROOT" logs "$OUT_ROOT/shared"
export FAULT_LOG_PATH="$OUT_ROOT/shared/faulted_request_ids.jsonl"
export LAUNCHER_STATUS_PATH="$OUT_ROOT/shared/launcher_status.json"
export MODEL

echo "=== [T1] fault-model unit tests ==="
pytest -q tests/test_fault_model_fid01.py

echo "=== [T2] start patched server (mode=$FAULT_MODE) ==="
python -m injections.launchers.launcherFID01 \
    --model "$MODEL" \
    --tensor-parallel-size "$TP" \
    --enable-per-request-metrics \
    --port 8000 \
    > "logs/server_${FAULT_MODE}_${FAULT_P}_${FAULT_SEED}.log" 2>&1 &
SRV_PID=$!
trap 'echo "killing server $SRV_PID"; kill $SRV_PID 2>/dev/null || true' EXIT

echo "=== waiting for /health ==="
for i in $(seq 120); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -sf http://localhost:8000/health >/dev/null || {
  echo "ERROR: server failed to start; see logs/server_*.log"; exit 1
}

echo "=== [T2] injection integration tests ==="
pytest -q tests/test_injection_e2e.py

echo "=== [T3] sweep (prefix-caching × 1 mode) ==="
for pc in on off; do
  OUT="${OUT_ROOT}/tp${TP}_${FAULT_MODE}_p${FAULT_P}_seed${FAULT_SEED}_prefix_${pc}"
  echo "--- run: $OUT ---"
  python runner.py \
    --out "$OUT" \
    --model "$MODEL" \
    --prefix-caching "$pc" \
    --tp "$TP"
done

echo "=== DONE $(date -u) ==="