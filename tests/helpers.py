from collections import Counter
from vllm.v1.core.block_pool import BlockPool

def make_pool():
  '''
  Creates a pool of 16 blocks.
  '''
  return BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=16) # match upstream args

def find_violations(live_requests):
  '''
  Invariant: a block's reference count >= number of live requests holding it. There is
  a mismatch if the pool thinks fewer requests use the block than actually do.
  args:
    live_requests: A dictionary of the requests currently running.
      e.g. {"A": [block1, block2, block3], "B": [block1, block2]}
  return:
    A sorted list of block IDs whose ref_cnt is lower than the number of live requests
    holding them. Empty if the invariant holds.
  '''
  holders = Counter(b.block_id for blocks in live_requests.values() for b in blocks)
  by_id = {b.block_id: b for blocks in live_requests.values() for b in blocks}
  return sorted(block_id for block_id, n in holders.items() if n > by_id[block_id].ref_cnt)

def find_negative_ref_counts(block_pool):
  '''
  Invariant: a block's reference count is never negative, meaning there was no more than
  one free.
  args:
    block_pool: All the blocks within a pool the requests have access to.
  return:
    A sorted list of block IDs whose ref_cnt is negative. Empty if the invariant holds.
  '''
  return sorted(block.block_id for block in block_pool.blocks if block.ref_cnt < 0)

def find_leaks(live_requests, block_pool):
  '''
  Invariant: Every block with a nonzero count is held by a live request.
  args:
    live_requests: A dictionary of the requests currently running.
      e.g. {"A": [block1, block2, block3], "B": [block1, block2]}
    block_pool: All the blocks within a pool the requests have access to.
  return:
    A sorted list of block IDs whose ref_cnt is positive but no live requests hold it.
    Empty if the invariant holds.
  '''
  held = {block.block_id for blocks in live_requests.values() for block in blocks}
  return sorted(block.block_id for block in block_pool.blocks
                if block.ref_cnt > 0 and block.block_id not in held)

def scope(live_requests, bad_ids):
  '''
  Given all the bad block IDs, finds the names of every request whose block list contains one
  of them.
  args:
    live_requests: The live requests holding blocks.
    bad_ids: The block IDs whose invariants have a mismatch.
  return:
    The affected requests.
  '''
  return {request_ids for request_ids, blocks in live_requests.items()
          if any(b.block_id in bad_ids for b in blocks)}
