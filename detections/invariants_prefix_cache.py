# invariants_prefix_cache.py
"""
Prefix-cache invariants: detection logic only.

Subsystem: 	prefix_cache
Process:    EngineCore (scheduler / KVCacheManager live here)
When:       consistency points - the boundary immediately after Scheduler.schedule()
            returns (i.e. after the scheduler has finished its state update and
            before the model runs).
Runs with:  always on, same configuration in clean and injected runs.

This module contains only the detection logic for prefeix cache fault detection. It
does not register a vLLM plugin, patch, nor import any vLLm symbols at module scope.
Please see /detection/invariants_plugin.py for that implementation.

This detector is blind to injections. It never reads the env var FAULT_MODE, never
imports the injection module, and organizes checks by subsystem, not FIDs. Several
faults may trip the same invariant detector; that is expected.

prefix_cache invariant catalogue:
	PC01 - resume point is block-aligned.
    PC02 - resume point agrees with the cached-block-count.
    PC03 - prefix-cache key matches the block it points at.
    PC04 - cached blocks are pinned.

TODO: move shared detection primitives (ViolationEvent, ViolationSink) into invariants_common.py
rather than importing across detectors once a second subsystem detector is developed.
"""
from __future__ import annotations
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Evidence record and sink (subsystem-agnostic primitives)
@dataclass
class ViolationEvent:
    """
    One detection-layer violation.

    Fields follow the detection contract: which chekc failed, where, when, and an
    evidence snapshot. Detection records what it saw; it does not name a fault class.
    Extra fields in 'evidence' are there forf classifaction use later.
    """
    check_pc_id: str = "" # e.g. PC01 .. PC04
    subsystem: str = "prefix_cache"
    step: int = -1 # scheduler step boundary
    timestamp: float = 0.0
    request_id: Optional[str] = None
    block_id: Optional[int] = None # only for per-block checks
    rank: Optional[int] = None # metadata checks run in EngineCore
    evidence: dict = field(default_factory=dict)
    detail: str = ""

    def to_json(self):
        """
        Logs the evidence snapshot into a JSON file.
        """
        return json.dumps(asdict(self), separators=(",",":"), default=str)

class ViolationSink:
    """
    Append-only JSONL sink.
    Detection write; classification reads.
    """
    def __init__(self, path: Optional[str] = None):
        self._path = path or os.environ.get(
            "DETECTION_LOG_PATH",
            os.path.join(os.getcwd(), "detection_events.jsonl"),
        )
        self._lock = threading.Lock()
        self._fh = None

    def emit(self, evidence: ViolationEvent):
        line = evidence.to_json()
        with self._lock:
            if self._fh is None:
                self._fh = open(self._path, "a", buffering=1)
            self._fh.write(line + "\n")

    def close(self):
        with self._lock:
            if self._fh is not none:
                self._fh.close()
                self._fh = None

# Small helpers
def _block_id(block):
    """
    Accept either a KVCacheBlock or a raw block id.
    args:
        block: Block given to be accepted.
    """
    return getattr(block, "block_id", block)

def _hash_value(hash):
    """
    Accept either a BlockHash object or a raw hash value.
    args:
        hash_val: Hash given to be accepted.
    """
    return getattr(hash_val, "hash_value", hash_val)

def _sample(sequence, k=8):
    """
    Keep the evidence snapshot small even for very long prompts.
    args:
        sequence: A list of evidence snapshots.
        k:
    """
    if len(sequence) <= 2 * k:
        return list(sequence)
    return list(sequence[:k] + list(seq[-k:]))

def _block_size_of(kv_manager):
    """
    Block size the KV Cache manager is actually using.

    Exposed for the plugin to pass into PreficCacheDetector at lazy init.
    """
    config = getattr(kv_manager, "kv_cache_config", None)
    groups = getattr(config, "kv_cache_groups", None) if config is not None else None
    if groups:
        size = {grp.kv_cache_spec.block_size for grp in groups}
        if len(sizes) == 1:
            return int(sizes.pop())
    block_size = getattr(kvm, "block_size", None)
    if block_size:
        return int(block_size)
    raise RuntimeError("invariant_prefix_cache: cannot determine block size.")

# Detector
class PrefixCacheDetector:
    """
    Runs prefix-cache invariants at EngineCore step boundaries.

    Hot path (the plugin's wrapper around get_computed_blcoks) calls
    note_lookup() - one list append, no I/O. All checking happens in
    on_step_boundary(), after the scheduler has finished its update.
    """
    SUBSYSTEM = "prefix_cache"

    def __init__(self, kv_manager, block_size,
                 sink=None, rank=None, state_check_interval=1):
        self.kv_manager = kv_manager
        self.block_size = int(block_size)
        self.sink = sink or ViolationSink()
        self.rank = rank
        self.state_check_interval = max(1, int(state_check_interval))
        self._step = 0
        self._pending = [] # list[tuple(Any list, int)]

    # hot path: buffer only
    # if there is a prefix-cache hit, record the blocks
    def note_lookup(self, request, blocks, n_computed):
        """
        Record the (blocks, n_computed) pair the scheduler will act on.

        Does no checking and no I/O. A cache miss (([], 0)) is not interesting
        for these invariants, so it is dropped here.
        """
        if request is None:
            return
        blocks = list(blocks) if blocks is not None else []
        if not blocks and n_computed <= 0:
            return
        self._pending.append((request, blocks, int(n_computed)))

    # consistency point
    def on_step_boundary(self):
        """
        Run all prefix-cache checks for this step.

        PC01/PC02 drain the buffered lookups (always run - cheap and the signal
        for the resume-point faults). PC03/PC04 inspect live hash-map state; they
        are gated by state_check_interval so the latency/overhead trade-off stays
        tunable.
        """
        try:
            self._check_resume_points()
        finally:
            self._pending.clear()

        if self._step % self.state_check_interval == 0:
            self._safe(self._check_hash_map)
            self._safe(self._check_cache_pinning)

        self._step += 1

    def _safe(self, fn):
        """
        Detection must never break serving. A vLLM internal rename should cost
        coverage, not an outage.
        """
        try:
            fn()
        except Exception:
            pass

    # PC01 / PC02: resume point
    def _check_resume_points(self):
        for request, blocks, n_computed in self._pending:
            self._check_pc1(request, n_computed)
            self._check_pc2(request, blocks, n_computed)

    def _check_pc1(self, request, n_computed):
        block_size = self.block_size
        remainder = n_computed % block_size
        if remainder == 0:
            return
        self._emit(ViolationEvent(
            check_pc_id="PC01",
            step=self._step,
            timestamp=time.time()
            request_id=getattr(request, "request_id", None),
            rank=self.rank,
            evidence={
                "n_computed": n_computed,
                "block_size": block_size,
                "remainder": remainder,
                "num_prompt_tokens": len(getattr(request, "prompt_token_ids", None) or []),
            },
            detail=(
                f"prefix-cache resume point {n_computed} is not block-aligned "
                f"(block_size)={block_size}, remainder={remainder}"
            ),
        ))

    def _check_pc2(self, request, blocks, n_computed):
        block_size = self.block_size
        n_blocks = len(blocks)
        expected = n_blocks * block_size
        if expected == n_computed:
            return
        delta = n_computed - expected
        direction_hint = "over_aligned" if delta > 0 else "under_aligned"
        self._emit(ViolationEvent(
            check_pc_id="PC02",
            step=self._step,
            timestamp=time.time()
            request_id=getattr(request, "request_id", None),
            rank=self.rank,
            evidence={
                "n_computed": n_computed,
                "n_blocks": n_blocks,
                "block_size": block_size,
                "expected_from_blocks": expected,
                "delta_tokens": delta,
                "direction_hint": direction_hint, # descriptive only; not a fault classification
                "block_ids": _sample([_block_id(block) for block in blocks]),
            },
            detail=(
                f"resume point {n_computed} != len(blocks)*block_size "
                f"({n_blocks}*{block_size}={expected}); {direction_hint} by "
                f"{abs(delta)} tokens"
            ),
        ))

    # PC03 / PC04: hash-map structure
    def _get_hash_map(self):
        """
        Attribute name has moved across vLLM v1 versions; take the first dict we find
        among the plausible ones.
        """
        for name in (
            "cached_block_hash_to_block",
            "block_hash_to_block",
            "cached_blocks",
            "prefix_cache",
        ):
            m = getattr(self.kv_manager, name, None)
            if isinstance(m, dict):
                return m
        return None

    def _check_hash_map(self):
        hash_map = self._get_hash_map()
        if not hash_map:
            return
        for key, block in list(hash_map.items()):
            block_hash = getattr(block, "block_hash", None)
            if block_hash is None:
                continue # block does not carry a hash; nothing to compare
            if _hash_value(block_hash) != _hash_value(key):
                self._emit(ViolationEvent(
                    check_pc_id="PC03",
                    step=self._step,
                    timestamp=time.time(),
                    block_id=_block_id(block),
                    rank=self.rank,
                    evidence={
                        "map_key_hash": repr(_hash_value(key)),
                        "block_hash": repr(_hash_value(block_hash)),
                    },
                    detail=(
                        "prefix-cache entry key does not match the block's recorded hash"
                    )
                ))

    def _check_cache_pinning(self):
        hash_map = self._get_hash_map()
        if not hash_map:
            return
        for key, block in list(hash_map.items()):
            ref = getattr(block, "ref_cnt", None)
            if ref is None:
                continue
            if ref < 1:
                self._emit(ViolationEvent(
                    check_pc_id="PC04",
                    step=self._step,
                    timestamp=time.time(),
                    block_id=_block_id(block),
                    rank=self.rank,
                    evidence={
                        "ref_count": ref,
                        "map_key_hash": repr(_hash_value(key)),
                    },
                    detail=(
                        f"cached block {_block_id(block)} has ref_count={ref}; "
                        "prefix-cache-pinned blocks must hold a reference"
                    )
                ))

    def _emit(self, evidence):
        try:
            self.sink.emit(evidence)
        except Exception:
            pass