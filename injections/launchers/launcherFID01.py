# injections/launchers/launcherFID01.py
import hashlib
import json
import os
import threading

FAULT_MODE = os.environ.get("FAULT_MODE", "clean")
FAULT_SEED = os.environ.get("FAULT_SEED", "0")
FAULT_P = float(os.environ.get("FAULT_P", "0.2"))
BLOCK_SIZE = 64 # vLLM's v2 block size TODO: Use vllm's config if possible

_FAULT_LOG_PATH = os.environ.get(
    "FAULT_LOG_PATH",
    os.path.join(os.getcwd(), "fault_request_ids.jsonl")
)
_STATUS_PATH = os.environ.get(
    "LAUNCHER_STATUS_PATH",
    os.path.join(os.getcwd(), "launcher_status.json")
)
_FAULT_LOG = open(_FAULT_LOG_PATH, "a", buffering=1)
_FAULT_LOG_LOCK = threading.Lock()

def _should_inject(request_id):
    h = int(hashlib.md5(f"{FAULT_SEED}:{request_id}".encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < FAULT_P

def _log_fault(request_id, clean, corrupted):
    with _FAULT_LOG_LOCK:
        _FAULT_LOG.write(json.dumps({
            "request_id": request_id,
            "clean": clean,
            "corrupted": corrupted,
        }) + "\n")
        _FAULT_LOG.flush()

# Patch only if a real mode is selected
import vllm.v1.core.kv_cache_manager as kvm
from injections.injectFID01 import Fid01Mode, inject_fid01

MODE_MAP = {
    "over_aligned": Fid01Mode.OVER_ALIGNED,
    "under_aligned": Fid01Mode.UNDER_ALIGNED,
    "misaligned": Fid01Mode.MISALIGNED,
}
MODE = MODE_MAP.get(FAULT_MODE)
MAGNITUDE = None # None -> per-mode default

if MODE is not None:
    _original = kvm.KVCacheManager.get_computed_blocks

    def _injected(self, *args, **kwargs):
        request = args[0] if args else kwargs.get("request")
        req_id = getattr(request, "request_id", None)
        result = _original(self, *args, **kwargs)
        blocks, n_computed = result

        if req_id is None:
            # Cannot attribute, do not inject
            return blocks, n_computed

        if (n_computed > 0
            and n_computed % BLOCK_SIZE == 0
            and _should_inject(req_id)):
            corrupted = inject_fid01(n_computed, BLOCK_SIZE, MODE, MAGNITUDE)
            _log_fault(req_id, n_computed, corrupted)
            return blocks, corrupted
        return blocks, n_computed

    kvm.KVCacheManager.get_computed_blocks = _injected

with open(_STATUS_PATH, "w") as f:
    json.dump({
        "active": MODE is not None,
        "mode": FAULT_MODE,
        "p": FAULT_P,
        "seed": FAULT_SEED,
        "pid": os.getpid(),
        "log_path": _FAULT_LOG_PATH,
    }, f)
print(
    f"[FID01] mode={FAULT_MODE} active={MODE is not None} "
    f"p={FAULT_P} seed={FAULT_SEED} log={_FAULT_LOG_PATH}",
    flush=True,
)

# Handing off to the real server entrypoint
from vllm.entrypoints.openai.api_server import main

if __name__ == "__main__":
    main()