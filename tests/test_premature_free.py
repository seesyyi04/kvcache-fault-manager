import helpers

def test_clean_run_no_violations():
  pool = helpers.make_pool()
  live = {"A": pool.get_new_blocks(3), "B": pool.get_new_blocks(3)}
  assert helpers.find_violations(live) == []
  
def test_premature_free():
  # request A
  pool = helpers.make_pool()
  a = pool.get_new_blocks(3)
  victim = a[1]
  
  # injected fault
  pool.free_blocks([victim])
  
  # request B
  b = pool.get_new_blocks(pool.get_num_free_blocks())
  live = {"A": a, "B": b}
    
  # fault manifested
  assert victim.block_id in {x.block_id for x in b}
  detect_bad = helpers.find_violations(live)
  print(f"\nInjected: block {victim.block_id}; Detected: {detect_bad}; Scope: {sorted(helpers.scope(live, detect_bad))}")
  # Is the block detected?
  assert detect_bad == [victim.block_id]
  # What is the scope?
  assert helpers.scope(live, detect_bad) == {"A", "B"}