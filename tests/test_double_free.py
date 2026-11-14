import helpers

def test_clean_run_no_violations():
  pool = helpers.make_pool()
  live = {"A": pool.get_new_blocks(3), "B": pool.get_new_blocks(3)}
  assert helpers.find_negative_ref_counts(pool) == []

def test_double_free():
  # request A
  pool = helpers.make_pool()
  a = pool.get_new_blocks(3)

  # injected fault
  pool.free_blocks(a)
  pool.free_blocks(a)

  # fault manifested
  detect_bad = helpers.find_negative_ref_counts(pool)
  print(f"\nInjected: double free of {[block.block_id for block in a]}; Detected: {detect_bad};")
  # Is the block detected?
  assert detect_bad == sorted(block.block_id for block in a)

  # What is the scope (next request)?
  try:
    pool.get_new_blocks(pool.get_num_free_blocks())
    output = "Allocation succeeded (silent)"
  except Exception as e:
    output = f"Allocation failed with {type(e).__name__}"
  print(f"Next allocation: {output}")
