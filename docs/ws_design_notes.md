# WS (weight-stationary) design notes — spec deviations and verified predictions

Companion to the WS spec ("weight-stationary PE + LOAD_WEIGHTS controller
state"). Records where the implementation deviated from the spec text, why,
and what the testbenches measured against the written-down predictions.
All deviations below were confirmed with the human before implementation.

## Resolved spec ambiguities / deviations

### 1. MAC staging: §3.2 vs §3.4 contradiction → 2-stage (registered product)

§3.2's mux equation (`sum_reg <= sum_in + (a * w_reg)`) implies a
combinational product; §3.4 asks for mac_unit's registered-product 2-stage
discipline. **Chosen: 2-stage** — `product <= a_in * w_reg` (the
just-arrived `a`, per §3.1's forwarding subtlety), then
`sum_reg <= b_in + product`. The systolic timing still aligns with the
existing 1-cycle-per-row `input_skew` because the psum and the product both
advance one row per cycle. Cost: one extra fill cycle.

### 2. INIT(clear) state dropped from the WS controller

§4's FSM sketch starts `INIT(clear)`. The WS PE has no persistent
accumulator to clear (`sum_reg` is a flow-through pipeline register), and
post-reset X in the sum path is covered by valid gating, exactly like the
skew pipes. FSM is `W_IDLE → W_LOAD → W_STREAM → W_FLUSH`.

### 3. Weight capture (§3.3 open item): broadcast capture, no per-PE compare

Neither the row-indexed capture-enable nor the `load_cycle_cnt == row_idx`
fallback was needed. Feeding tile rows **bottom-first** into the parallel
column shift chains aligns every row's weight in its own column
simultaneously (row r's word, fed at cycle `N-1-r`, sits in row r exactly
at cycle N), so a single broadcast `w_capture` pulse latches the whole
array at once. No diagonal load pattern required — columns are independent
shift registers.

### 4. Step 4's staging double-buffer: dropped as obsolete-by-construction

The OS controller's ping-pong staging exists because OS FEED cannot start
until a K+1-cycle BRAM fetch completes. The WS `W_LOAD` state instead
consumes BRAM words at exactly the rate the BRAM delivers them: read for
row `K-1-t` issued at cycle `t`, lands at `t+1`, fed straight onto the top
edge. The entire BRAM fetch latency visible per tile is the **1-cycle
lead-in** — that is all a staging bank could ever hide. The remaining LOAD
cost is the shift itself, which shares the `sum_reg` path with in-flight
psums and can never overlap STREAM without a per-PE shadow weight register
(considered and deferred; see below). Verdict: 2×N×K bytes of staging
registers + a prefetch sequencer to save 1 cycle in ~15 of per-tile
overhead is not worth the hardware. Step 4 was revised (human-approved) to
the back-to-back-tile TB case + this note.

**Deferred follow-up:** per-PE shadow `w_reg` (TPU-style) is the only
mechanism that actually hides the shift-in. Worth revisiting if perf
counters on real workloads show tile-swap overhead dominating (many small
tiles, short activation streams).

## Verified cycle budgets (N = K = 4, tb_ws_controller)

| Phase | Predicted | Measured |
|---|---|---|
| W_LOAD (shift + capture + BRAM lead-in) | N+2 = 6 | 6 (spec guessed ~2N) |
| W_STREAM (n_vecs + landing lead-in) | n_vecs+1 | 2 for n_vecs=1 |
| W_FLUSH (last psum exits) | 2N+1 = 9 | 9 — last column's result lands on the final flush cycle |
| 1-vector tile, total busy | 17 | 17 |
| Back-to-back tiles (2×1-vector) | 34 busy + 1 idle | 34 busy |

Per-tile swap overhead is **FLUSH 9 + LOAD 6 = 15 cycles**, independent of
`n_vecs` — amortized by streaming more vectors per resident tile, which is
the batching story (out of scope here).

## Conventions

- Result: bottom of column j yields `sum_i w[i][j] * a[i]`, i.e.
  `y = W_loaded^T · a`. Hosts store weight tiles **transposed** to compute
  `y = W·a`.
- Weight tile in the buffer: word `i` (at `tile_idx*K + i`) = resident row
  `i`, lane `j` (bits `[8j+7:8j]`) = column `j`.
- Activation word: lane `i` feeds array row `i`; one word = one vector.
- Column outputs are skewed (column j lags column 0 by j cycles) and exit
  continuously, gated by `sum_valid` — no drain. Consumers deskew.
