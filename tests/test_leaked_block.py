import helpers

def test_clean_run_no_violations():
  pool = helpers.make_pool()
  live = {"A": pool.get_new_blocks(3), "B": pool.get_new_blocks(3)}
  assert helpers.find_leaks(live, pool) == []

def test_leaked_block():
  pool = helpers.make_pool()
  free_before = pool.get_num_free_blocks()

  a = pool.get_new_blocks(3)
  leaked_block = a[1]

  # injected fault
  pool.free_blocks([block for block in a if block is not leaked_block])
  live = {}

  detect_bad = helpers.find_leaks(live, pool)
  print(f"\nInjected: leak of block {leaked_block.block_id}; Detected {detect_bad}")
  assert detect_bad == [leaked_block.block_id]
  assert pool.get_num_free_blocks() == free_before - 1
