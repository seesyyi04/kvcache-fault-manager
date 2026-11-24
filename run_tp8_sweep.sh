#!/usr/bin/env bash
# FID01 sweep at one TP size for one or more models.
#
# Per model, 6 runs, each with a FRESH server (same empty-cache starting state):
#   clean          prefix on    (baseline)
#   clean_repeat   prefix on    (noise floor: diff against baseline)
#   clean          prefix off   (FID01 reference: prefix-caching-disabled run)
#   under_aligned  prefix on
#   over_aligned   prefix on
#   misaligned     prefix on
# Fault modes are not run with prefix caching off: FID01 only fires on prefix-cache hits.
#
# Usage:
#   bash run_tp8_sweep.sh                       # both models, TP=8
#   MODELS="meta-llama/Llama-3.1-8B-Instruct" bash run_tp8_sweep.sh
#   DRY_RUN=1 bash run_tp8_sweep.sh             # print what would run, start nothing
#
# Requires:
#   - this repo installed in the same environment as vLLM (uv pip install -e .), so vLLM
#     loads the FID01 plugin; the plugin reads FAULT_MODE / FAULT_P / FAULT_SEED /
#     FAULT_LOG_PATH / LAUNCHER_STATUS_PATH from the server's environment
#   - runner.py is a client of the server on $PORT and writes <out>/outputs.jsonl
# Check SERVER_FLAGS against `vllm serve --help` before the first run.
# Re-running skips any run whose outputs.jsonl already exists (resume after a crash).

set -euo pipefail

MODELS="${MODELS:-meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-70B-Instruct}"
TP="${TP:-8}"
FAULT_P="${FAULT_P:-0.2}"
FAULT_SEED="${FAULT_SEED:-0}"
PORT="${PORT:-8000}"
OUT_ROOT="$(realpath -m "${OUT_ROOT:-runs}")"
LOG_DIR="$(realpath -m "${LOG_DIR:-logs}")"
DRY_RUN="${DRY_RUN:-0}"
DOWNLOAD="${DOWNLOAD:-1}"          # 1 = make sure each model is downloaded first
CONCURRENCY="${CONCURRENCY:-64}"   # requests in flight per run
# Extra server flags. Remove any that `vllm serve --help` doesn't list; add
# --enable-per-request-metrics here if your vLLM has it.
SERVER_FLAGS="${SERVER_FLAGS:---enable-prompt-tokens-details --enable-request-id-headers}"

# name | fault mode | prefix caching | fault probability
RUNS=(
  "clean|clean|on|0"
  "clean_repeat|clean|on|0"
  "clean|clean|off|0"
  "under_aligned|under_aligned|on|${FAULT_P}"
  "over_aligned|over_aligned|on|${FAULT_P}"
  "misaligned|misaligned|on|${FAULT_P}"
)

mkdir -p "$OUT_ROOT" "$LOG_DIR"
SRV_PID=""

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

health_timeout_for() {
  # 70B needs several minutes to load 140 GB and compile; 8B is much faster.
  case "$1" in
    *70B*) echo "${HEALTH_TIMEOUT_70B:-1500}" ;;
    *)     echo "${HEALTH_TIMEOUT_8B:-600}" ;;
  esac
}

stop_server() {
  if [[ -n "$SRV_PID" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
    log "stopping server (process group $SRV_PID)"
    kill -- -"$SRV_PID" 2>/dev/null || true
    for _ in $(seq 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$SRV_PID" 2>/dev/null || true
  fi
  SRV_PID=""
  # wait until the GPUs are actually free before the next start
  if command -v nvidia-smi >/dev/null && [[ "$DRY_RUN" != 1 ]]; then
    for _ in $(seq 120); do
      [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && return 0
      sleep 1
    done
    log "WARNING: GPU processes still present after stop:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  fi
}
trap stop_server EXIT

start_server() {  # model, prefix on|off, server log path
  local model="$1" pc="$2" srv_log="$3"
  local pc_flag="--enable-prefix-caching"
  [[ "$pc" == off ]] && pc_flag="--no-enable-prefix-caching"

  # shellcheck disable=SC2206  # SERVER_FLAGS is split into words on purpose
  local cmd=(vllm serve "$model" --tensor-parallel-size "$TP" "$pc_flag" --port "$PORT"
             $SERVER_FLAGS)
  log "start: FAULT_MODE=$FAULT_MODE FAULT_P=$FAULT_P ${cmd[*]}"
  [[ "$DRY_RUN" == 1 ]] && return 0

  setsid "${cmd[@]}" > "$srv_log" 2>&1 &
  SRV_PID=$!

  local timeout; timeout="$(health_timeout_for "$model")"
  local waited=0
  until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      log "ERROR: server exited during startup; last lines of $srv_log:"
      tail -n 30 "$srv_log"; return 1
    fi
    if (( waited >= timeout )); then
      log "ERROR: server not healthy after ${timeout}s; see $srv_log"; return 1
    fi
    sleep 5; waited=$((waited + 5))
  done
  log "server healthy after ${waited}s"
}

model_tag() {  # meta-llama/Llama-3.1-70B-Instruct -> llama3.1-70b
  local m="${1##*/}"; m="${m,,}"; m="${m/llama-/llama}"; echo "${m%-instruct}"
}

log "unit tests"
[[ "$DRY_RUN" == 1 ]] || pytest -q tests/test_fault_model_fid01.py

for model in $MODELS; do
  tag="$(model_tag "$model")"
  log "===== model $model ($tag), TP=$TP ====="

  if [[ "$DOWNLOAD" == 1 && "$DRY_RUN" != 1 ]]; then
    : "${HF_TOKEN:?set HF_TOKEN before running (needed to download gated models)}"
    log "ensuring $model is downloaded (skips files already present)"
    hf download "$model" > /dev/null
  fi

  for spec in "${RUNS[@]}"; do
    IFS='|' read -r name FAULT_MODE pc FAULT_P_RUN <<< "$spec"
    out="${OUT_ROOT}/${tag}/tp${TP}_${name}_p${FAULT_P_RUN}_seed${FAULT_SEED}_prefix_${pc}"

    if [[ -f "$out/outputs.jsonl" ]]; then
      log "skip (already done): $out"; continue
    fi
    mkdir -p "$out"

    # per-run ground truth, so injections from different runs never mix
    export FAULT_MODE FAULT_SEED MODEL="$model"
    export FAULT_P="$FAULT_P_RUN"
    export FAULT_LOG_PATH="$out/faulted_request_ids.jsonl"
    export LAUNCHER_STATUS_PATH="$out/launcher_status.json"
    srv_log="$LOG_DIR/${tag}_tp${TP}_${name}_prefix_${pc}.log"

    log "--- run: $out ---"
    start_server "$model" "$pc" "$srv_log"
    t0=$(date +%s)
    if [[ "$DRY_RUN" == 1 ]]; then
      log "would run: python runner.py --out $out --model $model --prefix-caching $pc --tp $TP --concurrency $CONCURRENCY"
    else
      python runner.py --out "$out" --model "$model" --prefix-caching "$pc" --tp "$TP" \
        --concurrency "$CONCURRENCY"
      cp "$srv_log" "$out/server.log"
      [[ "$FAULT_MODE" != clean && ! -s "$FAULT_LOG_PATH" ]] && \
        log "WARNING: $name produced no fault-log lines; check the plugin is installed and FAULT_MODE reached the server"
    fi
    log "run took $(( $(date +%s) - t0 ))s"
    stop_server
  done
done

log "=== DONE ==="