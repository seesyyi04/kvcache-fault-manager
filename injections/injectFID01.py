# injections/injectFID01.py
from enum import Enum, auto

class Fid01Mode(Enum):
    OVER_ALIGNED = auto() # +k whole blocks
    UNDER_ALIGNED = auto() # -k whole blocks
    MISALIGNED = auto() # +/- partial block

def check_aligned(resume_token_index, block_size):
    if block_size <= 0:
        raise ValueError(f"block_size must be positive; each block is {block_size}")
    if resume_token_index < 0:
        raise ValueError(f"resume_token_index must be >= 0; got {resume_token_index}")
    if resume_token_index % block_size != 0:
        raise ValueError(f"Correct resume point is not a multiple of block_size {block_size}")

def over_aligned(resume_token_index, block_size, k_blocks=1):
    """
    Resume k whole blocks after the correct point.
    Cache effect: blocks that should have been recomputed are reused from cache
    instead -> stale / missing KV for k*block_size tokens.
    """
    check_aligned(resume_token_index, block_size)
    if k_blocks < 1:
        raise ValueError(f"k_blocks must be >= 1")
    return resume_token_index + k_blocks * block_size

def under_aligned(resume_token_index, block_size, k_blocks=1):
    """
    Resume k whole blocks before the correct point.
    Cache effect: blocks that were already cached get recomputed -> wasteful but
    produces identical KV values (a "safe" fault).
    """
    check_aligned(resume_token_index, block_size)
    if k_blocks < 1:
        raise ValueError(f"k_blocks must be >= 1")
    return max(0, resume_token_index - k_blocks * block_size)

def misaligned(resume_token_index, block_size, offset=None):
    """
    Resume at a token index inside the correct block (not on a boundary).
    Cache effect: downstream code slices input_ids at this token but still
    derives the block table via floor division -> the "which blocks are
    cached" answer and "how many tokens" answer disagree.

    offset defaults to +block_size // 2 (mid-block).
    Must satisfy 0 < |offset| < block_size.
    """
    check_aligned(resume_token_index, block_size)
    if offset is None:
        offset = block_size // 2
    if not (0 < abs(offset) < block_size):
        raise ValueError(f"Offset must satisfy 0 < |offset| < {block_size}; offset got {offset}")
    return resume_token_index + offset

def inject_fid01(resume_token_index, block_size, mode, magnitude=None):
    """
    Dispath fault injection.
    magnitude:
        OVER_ALIGNED / UNDER_ALIGNED -> k_blocks (default=1)
        MISALIGNED -> offset in tokens (default=(block_size//2))
    """
    if mode is Fid01Mode.OVER_ALIGNED:
        return over_aligned(resume_token_index, block_size,
                            magnitude if magnitude is not None else 1)
    if mode is Fid01Mode.UNDER_ALIGNED:
        return under_aligned(resume_token_index, block_size,
                            magnitude if magnitude is not None else 1)
    if mode is Fid01Mode.MISALIGNED:
        return misaligned(resume_token_index, block_size, offset=magnitude)
    raise ValueError(f"InjectFID01: Unknown fault mode: {mode}")