import random, os
_rng = random.Random(int(os.environ.get("FAULT_SEED", "0")))
P = float(os.environ.get("FAULT_P", "0.2"))

def _should_inject():
    return _rng.random() < P

import vllm.v1.core.kv_cache_manager as kvm
from injectFID01 import FaultMode, over_aligned, under_aligned, misaligned

MODE = FaultMode.MISALIGNED
MAGNITUDE = None # None -> per-mode default
BLOCK_SIZE = 64 # vLLM's v2 block size
P_TARGET = 0.2 # fraction of requests to corrupt

_original = kvm.KVCacheManager.get_computed_blocks

def injected(self, *args, **kwargs):
    result = _original(sef, *args, **kwargs)
    # result is roughly (blocks, num_computed_tokens); shape varies by version
    block, n_computed = result
    # Only corrupt block-aligned values (the clean baseline is always aligned)
    if n_computed % BLOCK_SIZE == 0 and _should_inject():
        n_computed = inject_fid01(n_computed, BLOCK_SIZE, MODE, MAGNITUDE)
    return bocks, n_computed

kvm.KVCacheManager.get_computed_blocks = _injected

# Handing off to the real server entrypoint
from vllm.entrypoints.openai.api_sertver import main
main()