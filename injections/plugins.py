"""
vLLM general plugin
TODO: generalize for future FIDs

Registered in pyproject.toml under [project.entry-points."vllm.general_plugins"].
vLLM calls register() in each process it starts (API server, engine core, GPU workers).
Only the engine-core process ever calls KVCacheManager.get_computed_blocks, so only
that process writes the fault log and the status file.

Environment variables (read once, when the patch is installed)
    FAULT_MODE -> clean | over_aligned | under_aligned | misaligned (clean = no patch)
    FAULT_P -> probabiloty of injecting on an eligible hit (default = 0.2)
    FAULT_SEED -> seed for choosing which prompts get injected (default = 0)
    FAULT_MAGNITUDE -> k_blocks for over/under, offset tokens for misaligned (default per mode)
    FAULT_LOG_PATH -> JSONL log: one line per eligible prefix-cache hit
    LAUNCHER_STATUS_PATH -> JSON status, written on the first hit (proves the patch fired)
"""
import hashlib
import json
import os
import threading
import time

_PATCHED = False
_lock = threading.Lock()
_log = None
_status_written = False

_ORIGINAL = None
_CFG = None

def register():
    """
    Called by vLLM in every process it starts.
    """
    global _PATCHED
    if _PATCHED or os.environ.get("FAULT_MODE", "clean") == "clean":
        return
    _install()
    _PATCHED = True

def _block_size_of(mgr):
    """
    Block size the KV cache manager is actually using.
    """
    cfg = getattr(mgr, "kv_cache_config", None)
    groups = getattr(cfg, "kv_cache_groups", None) if cfg is not None else None
    if groups:
        sizes = {g.kv_cache_spec.block_size for g in groups}
        if len(sizes) != 1:
            raise RuntimeError(f"FID01 assumes one KV block size, found {sorted(sizes)}")
        return sizes.pop()
    bs = getattr(mgr, "block_size", None)
    if bs:
        return bs
    raise RuntimeError("FID01: could not read block size from KVCacheManager")

def _stable_key(request):
    """
    Same prompt -> same key in every run (server request IDs are random).
    """
    toks = getattr(request, "prompt_token_ids", None)
    if toks is not None:
        return repr(tuple(toks))
    return str(getattr(request, "request_id", ""))

def _should_inject(request, seed, p):
    h = int(hashlib.md5(f"{seed}:{_stable_key(request)}".encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < p

def _write(record):
    global _log
    with _lock:
        if _log is None:
            path = os.environ.get("FAULT_LOG_PATH",
                                  os.path.join(os.getcwd(), "fault_request_ids.jsonl"))
            _log = open(path, "a", buffering=1)
        _log.write(json.dumps(record) + "\n")

def _write_status_once(mode_name, p, seed, magnitude, block_size):
    global _status_written
    if _status_written:
        return
    path = os.environ.get("LAUNCHER_STATUS_PATH",
                          os.path.join(os.getcwd(), "launcher_status.json"))
    with open(path, "w") as f:
        json.dump({"active": True, "mode": mode_name, "p": p, "seed": seed,
                   "magnitude": magnitude, "pid": os.getpid(),
                   "block_size": block_size}, f)
    _status_written = True

def injected(self, *args, **kwargs):
    """
    Replacement for KVCacheManager.get_computed_blocks.
    """
    request = args[0] if args else kwargs.get("request")
    blocks, n_computed = _ORIGINAL(self, *args, **kwargs)
    if request is None or n_computed <= 0:
        return blocks, n_computed # not a prefix-cache hit

    cfg = _CFG
    block_size = _block_size_of(self)
    _write_status_once(cfg["mode_name"], cfg["p"], cfg["seed"], cfg["magnitude"], block_size)

    record = {
        "t": time.time(),
        "pid": os.getpid(),
        "request_id": getattr(request, "request_id", None),
        "mode": cfg["mode_name"],
        "block_size": block_size,
        "num_tokens": getattr(request, "num_tokens", None),
        "num_prompt_tokens": len(getattr(request, "prompt_token_ids", None) or []),
        "clean": n_computed,
        "injected": False,
        "corrupted": None,
        "skip_reason": None,
    }

    if n_computed % block_size != 0:
        record["skip_reason"] = "hit_not_block_aligned"
    elif not _should_inject(request, cfg["seed"], cfg["p"]):
        record["skip_reason"] = "not_selected"
    else:
        try:
            corrupted = cfg["inject_fid01"](n_computed, block_size, cfg["mode"], cfg["magnitude"])
        except ValueError as e:
            record["skip_reason"] = f"fault_model_rejected: {e}"
        else:
            if corrupted == n_computed:
                record["skip_reason"] = "no_op"
            else:
                record.update(injected=True, corrupted=corrupted)
                _write(record)
                return blocks, corrupted
    _write(record)
    return blocks, n_computed

# The patch
def _install():
    global _ORIGINAL, _CFG
    import vllm.v1.core.kv_cache_manager as kvm # only imported when fault is active
    from injections.injectFID01 import Fid01Mode, inject_fid01

    mode_map = {
        "over_aligned": Fid01Mode.OVER_ALIGNED,
        "under_aligned": Fid01Mode.UNDER_ALIGNED,
        "misaligned": Fid01Mode.MISALIGNED,
    }
    mode_name = os.environ.get("FAULT_MODE", "clean")
    mode = mode_map.get(mode_name)
    if mode is None:
        raise ValueError(f"FID01: unknown FAULT_MODE={mode_name!r}; "
                         f"expected clean or one of {sorted(mode_map)}")
    p = float(os.environ.get("FAULT_P", "0.2"))
    seed = os.environ.get("FAULT_SEED", "0")
    mag_env = os.environ.get("FAULT_MAGNITUDE")
    magnitude = int(mag_env) if mag_env not in (None, "") else None

    _CFG = {"mode_name": mode_name, "mode": mode, "p": p, "seed": seed,
            "magnitude": magnitude, "inject_fid01": inject_fid01}
    _ORIGINAL = kvm.KVCacheManager.get_computed_blocks
    kvm.KVCacheManager.get_computed_blocks = injected
    print(f"[FID01] patch installed pid={os.getpid()} mode={mode_name} "
          f"p={p} seed={seed} magnitude={magnitude}", flush=True)
