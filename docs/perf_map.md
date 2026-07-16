# perf_counters register map (CONTRACT)

This map is read by Phase 3's AXI-Lite/Python. Do not renumber; add new
counters at the end. Mirrored in the `rtl/perf_counters.sv` header — keep
both in sync.

| addr | name             | semantics                                            |
|------|------------------|------------------------------------------------------|
| 0    | total_cycles     | every cycle since perf_clear was last released       |
| 1    | fetch_cycles     | cycles the controller was in S_FETCH                 |
| 2    | feed_cycles      | cycles the controller was in S_FEED                  |
| 3    | settle_cycles    | cycles the controller was in S_SETTLE                |
| 4    | drain_cycles     | cycles the controller was in S_DRAIN                 |
| 5    | idle_cycles      | cycles the controller was in S_IDLE                  |
| 6    | chunks_completed | count of chunk_done strobes (last FEED beat/chunk)   |
| 7    | tiles_completed  | count of tile_done strobes (last DRAIN cycle/walk)   |

Properties:

- **Width:** 32 bits (`CW` parameter; the map itself is width-agnostic).
- **Saturating:** counters hold at all-ones, never wrap.
- **Clear:** `perf_clear` zeroes all counters (that cycle does not count).
  With tiling_engine's `AUTO_CLEAR=1` (default), an accepted `start` also
  clears, so numbers are per-run; set `AUTO_CLEAR=0` for cumulative counts.
- **Read protocol:** present `perf_addr`, read `perf_data` one cycle later
  (registered read, same contract as tile_buffer). Out-of-range addresses
  (8-15) read 0.
- **Passive:** the counter block observes explicit Moore flags exported by
  the controller; no counter output feeds any control logic. S_INIT (one
  cycle after reset) is covered by no state flag.
- **Utilization definition used in perf_baseline.md:**
  `feed_cycles / total_cycles`.

## WS counter map (ws_perf_counters instance inside ws_unit)

Same hardware (`perf_counters`), WS categories. Read over AXI-Lite via
PERF_SEL (0x10) / PERF_DATA (0x14) in `axi_lite_ctrl`.

| idx | name          | meaning                                    |
|-----|---------------|--------------------------------------------|
| 0   | total_cycles  | every cycle since perf_clear release        |
| 1   | load_cycles   | cycles in W_LOAD (= N+2 per tile)           |
| 2   | stream_cycles | cycles in W_STREAM (= n_vecs+1 per tile)    |
| 3   | flush_cycles  | cycles in W_FLUSH (= 2N+1 per tile)         |
| 4   | (unused)      | reads 0                                     |
| 5   | idle_cycles   | cycles in W_IDLE                            |
| 6   | (unused)      | reads 0                                     |
| 7   | tile_swaps    | count of weight-tile loads (1 per go)       |

Indices 0/5/7 deliberately match the OS map (total/idle/tile count) so
host code can share offsets. Counters saturate; out-of-range reads 0.
No AXI perf_clear bit yet (counters count from core reset; hosts use
deltas) — first candidate for a CTRL bit when needed.
