import functools, os
os.environ["VLLM_ENABLE_V1_MULTIPLROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler

