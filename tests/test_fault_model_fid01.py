# tests/test_fault_model_fid01.py
import pytest
from injections.injectFID01 import (
    Fid01Mode, over_aligned, under_aligned, misaligned, inject_fid01,
)

BLOCK_SIZE = 64 # vLLM V2 default; the plugin reads the real value at runtime

"""
Over-aligned
"""
# Next block boundary
def test_over_returns_later_block():
    # Resume on the next block boundary.
    # Expected = 192
    assert over_aligned(BLOCK_SIZE*2, BLOCK_SIZE, k_blocks=1) == 192

# Boundary on two blocks later
def test_over_k_blocks():
    # Resume on the block boundary two blocks later.
    # Expected = 256
    assert over_aligned(BLOCK_SIZE*2, BLOCK_SIZE, k_blocks=2) == 256

# Non-aligned input (over)
def test_over_aligned_requires_block_aligned_input():
    with pytest.raises(ValueError, match="multiple"):
        over_aligned(37, BLOCK_SIZE) # 37 % 64 != 0

"""
Under-aligned
"""
# Previous block boundary
def test_under_returns_earlier_block():
    # Resume on the previous block boundary.
    # Expected = 64
    assert under_aligned(BLOCK_SIZE*2, BLOCK_SIZE, k_blocks=1) == 64

# Clamps at zero
def test_under_clamps_at_zero():
    # Resume at position zero of the first block.
    # Expected = 0
    assert under_aligned(BLOCK_SIZE, BLOCK_SIZE, k_blocks=1) == 0

# "Safe" condition; make sure under-alignment is calculated correctly
def test_under_aligned_property_is_monotone():
    # Under-aligned must always produce a value ≤ the input.
    for r in (0, 64, 128, 256, 512):
        for k in (1, 2, 3):
            assert under_aligned(r, BLOCK_SIZE, k_blocks=k) <= r

# Non-aligned input (under)
def test_under_aligned_requires_block_aligned_input():
    with pytest.raises(ValueError, match="multiple"):
        under_aligned(37, BLOCK_SIZE) # 37 % 64 != 0

def test_under_clamps_at_zero_from_absolute_zero():
    assert under_aligned(0, BLOCK_SIZE, k_blocks=1) == 0
    assert under_aligned(0, BLOCK_SIZE, k_blocks=5) == 0

"""
Misaligned
"""
def test_misaligned_default_offset_is_midblock():
    # Default offset is block_size // 2 = 32
    assert misaligned(BLOCK_SIZE * 2, BLOCK_SIZE) == 160

def test_misaligned_with_explicit_offset():
    # Offset = 1
    # Expected = resume_token_index + 1 = 128 + 1 = 129
    assert misaligned(BLOCK_SIZE*2, BLOCK_SIZE, offset=1) == 129

def test_misaligned_negative_offset():
    # Offset = 01 -> one token before the boundary
    assert misaligned(BLOCK_SIZE*2, BLOCK_SIZE, offset=-1) == 127

def test_misaligned_offset_out_of_range():
    r = BLOCK_SIZE * 2
    with pytest.raises(ValueError, match="offset"):
        misaligned(r, BLOCK_SIZE, offset=BLOCK_SIZE)
    with pytest.raises(ValueError, match="offset"):
        misaligned(r, BLOCK_SIZE, offset=-BLOCK_SIZE)
    with pytest.raises(ValueError, match="offset"):
        misaligned(r, BLOCK_SIZE, offset=0)

def test_misaligned_rejects_negative_result():
    # A negative offset at resume point 0 would give a negative token count.
    with pytest.raises(ValueError, match="negative"):
        misaligned(0, BLOCK_SIZE, offset=-1)

def test_misaligned_returns_unaligned():
    for r in (0, 64, 128, 256, 512, 1024):
        result = misaligned(r, BLOCK_SIZE)
        assert result % BLOCK_SIZE != 0, (
            f"misaligned({r}) returned aligned value {result} — "
            f"this is not a misalignment, it's an over- or under-alignment in disguise"
        )

"""
dispatch
"""
def test_dispatch_all_modes():
    r = BLOCK_SIZE * 2

    # Values match direct calls
    assert inject_fid01(r, BLOCK_SIZE, Fid01Mode.OVER_ALIGNED) == 192
    assert inject_fid01(r, BLOCK_SIZE, Fid01Mode.UNDER_ALIGNED) == 64
    assert inject_fid01(r, BLOCK_SIZE, Fid01Mode.MISALIGNED) == 160

    # Dispatch invariant: over/under preserve alignement, misaligned breaks it
    assert inject_fid01(r, BLOCK_SIZE, Fid01Mode.OVER_ALIGNED) % BLOCK_SIZE == 0
    assert inject_fid01(r, BLOCK_SIZE, Fid01Mode.UNDER_ALIGNED) % BLOCK_SIZE == 0
    assert inject_fid01(r, BLOCK_SIZE, Fid01Mode.MISALIGNED) % BLOCK_SIZE != 0

def test_dispatch_passes_magnitude_through():
    r = BLOCK_SIZE * 4
    assert inject_fid01(r, BLOCK_SIZE, Fid01Mode.OVER_ALIGNED, magnitude=2) == r + 2 * BLOCK_SIZE
    assert inject_fid01(r, BLOCK_SIZE, Fid01Mode.UNDER_ALIGNED, magnitude=3) == r - 3 * BLOCK_SIZE
    assert inject_fid01(r, BLOCK_SIZE, Fid01Mode.MISALIGNED, magnitude=5) == r + 5

def test_dispatch_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown fault mode"):
        inject_fid01(BLOCK_SIZE, BLOCK_SIZE, "over_aligned")