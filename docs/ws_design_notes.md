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

## Verified cycle budgets (tb_ws_controller at N=4; tb_ws_n16 at N=16)

General formulas, certified at BOTH sizes:
**LOAD = N+2, STREAM = n_vecs+1, FLUSH = 2N+1, busy = 3N+4+n_vecs.**

| Phase | Predicted | Measured N=4 | Measured N=16 |
|---|---|---|---|
| W_LOAD | N+2 | 6 (spec guessed ~2N) | 18 |
| W_STREAM | n_vecs+1 | 2 (nv=1) | 2 / 5 (nv=1 / 4) |
| W_FLUSH | 2N+1 | 9 | 33 |
| busy, 1-vector tile | 3N+5 | 17 | 53 |
| busy, 4-vector tile | 3N+8 | — | 56 |
| Back-to-back tiles (2×1-vec, N=4) | 34 busy + 1 idle | 34 busy | — |
| Output cadence peak | min(n_vecs, N) | 1 / 4 | 1 / 4 |

Per-tile swap overhead is **FLUSH + LOAD = 3N+3 cycles** (15 at N=4, 51 at
N=16), independent of `n_vecs` — amortized by streaming more vectors per
resident tile, which is the batching story (out of scope here).

## Synthesis: DSP48 inference (bug iterations, post-merge)

### Iteration 1: 0 DSPs → missing use_dsp attribute

First `ws_unit` synth came back **0 DSP**, `u_array` = 1176 LUT / 784 FF /
176 CARRY8 (~73 LUT/PE) — all arithmetic in fabric. Root cause: `ws_pe`
was missing the `(* use_dsp = "yes" *)` attribute that `mac_unit` carries
(its header: "necessary to cut down on LUTs"). Structural hypotheses
eliminated against the working OS reference first: datapath regs unreset
(same split), mode mux in the same always block as the arithmetic (same
shape as mac_unit's clear/drain/accumulate mux). Vivado's default
heuristic leaves 8×8 multiplies in LUT fabric — same natural cause as the
Phase 2 OS incident. Applied at module scope (commit c0f0a23).

**Prediction:** 16 DSP, LUT well under ~200, CARRY8 ≈ 0.
**Measured:** LUT 145 / FF 40 held, but **DSP = 32 (2/PE)** — module
scope forced the 32-bit add into a second DSP48 ALU instead of fusing.

### Iteration 2: 32 DSPs → attribute rescoped to the multiplier signal

Register-by-register diff against mac_unit found the template difference:

- mac_unit (fuses): runtime select on the **Z mux** —
  `acc <= drain_in` (P = C) vs `acc <= acc + product` (P = P + M),
  X/Y constant. This is Xilinx's documented loadable-MACC inference
  template.
- ws_pe: runtime select on the **X/Y side** —
  `sum_reg <= b_in` (P = C) vs `sum_reg <= b_in + product` (P = C + M),
  Z constant. Inverse of the recognized template.

Fix attempt (one variable, commit 9898d28): `use_dsp` moved from module
scope to the `product` signal only.

**Prediction:** DSP = 16 with LUT ≤ ~145 if fusion works; DSP = 16 with
LUTs RISING ⇒ fusion still blocked by the X/Y-side select.
**Measured: DSP = 16, LUT 353, FF 364, CARRY8 48** — the discriminator's
second arm: multiply in DSPs, 32-bit add + result mux + sum_reg all in
fabric. X/Y-side mode select confirmed as the fusion blocker.
(BRAM = 1 at unit level: the two 64×32 tile_buffers pack as 2 RAMB18 in
one RAMB36 tile — expected, not part of this bug.)

### Iteration 3: static P = C + M, gating moved into the product register

Stage 2 is now mux-free — `sum_reg <= b_in + product` unconditionally —
and all gating happens at stage 1:
`product <= valid_in ? (a_in * w_reg) : '0`.

- Cycle-exact equivalent to the old stage-2 gating
  (`valid_pipe(t) = valid_in(t-1)`, so both produce
  `sum_reg(t) = b_in + (valid_in(t-1) ? a(t-1)*w : 0)`).
- The LOAD passthrough arm is gone: during LOAD, product is provably 0
  because valid has been low ≥ 1 cycle (FLUSH is 2N+1 cycles). This is
  now a ws_pe CONTRACT, enforced by a sim-only tripwire (same hazard
  class as the OS bank-landing tripwire): valid activity during
  load_mode is a $error.
- The gate is a mux AFTER the multiply, not on the operand: `0 * X`
  is X in 4-state simulation (w_reg is X before the first capture) and
  would corrupt the first LOAD in sim; the chosen form is X-safe and
  maps to the DSP48 MREG's synchronous reset (RSTM = !valid).

**Prediction before re-synth:** DSP = 16 (fully fused: MREG = product
with RSTM, PREG = sum_reg, b_in on C, static OPMODE), LUT back down to
~145–200, CARRY8 ≈ 0, FF back near ~40 as sum_reg reabsorbs into P
registers. If DSP = 16 but LUT/FF stay at iteration-2 levels, the C-port
add still isn't packing and the next suspect is the b_in fan-in
(sum_reg also feeds the w_capture byte mux).

*(Measured: human reported "confirmed cost-per-lane" in the N=16 spec,
implying fusion confirmed — exact numbers still to be logged here.)*

## Requant integration (see reports/report-requant.md)

4 lanes per integration (OS: buffer-read tap in quant_engine; WS: live
bottom-edge tap in ws_quant_unit). Cadence measured: min(n_vecs, N)
columns valid simultaneously. Per-lane synth cost: the 32×16 scale
multiply exceeds one DSP48's 27×18 → predicted 2 DSP/lane (human
confirmed at N=4 per the N=16 spec).

## N=16 scaling checkpoint (verification spec, 2026-07-16)

Parameterization gap found and fixed (mechanical): buffer word width was
hardcoded 32-bit in ws_controller's rd_data ports, ws_unit's WORD_WIDTH +
write ports, ws_quant_unit's write ports; now N*DATA_WIDTH (one word =
one tile row = one activation vector; 128-bit at N=16). Everything else
was already generate-loop parameterized. N=4 defaults unchanged; full
N=4 regression untouched and green.

**Functional at N=16 (tb_ws_n16, measured):** raw 16×16 matvec exact,
identity-quant int8 + ReLU correct, busy = 53 (nv=1) and 56 (nv=4),
cadence peak = min(n_vecs, N). All formulas held — see cycle-budget
table above.

**Synthesis predictions (written before the human's N=16 synth run):**

| Quantity | Predicted N=16 | Basis | Measured |
|---|---|---|---|
| Array DSP | 256 (N²) | 1 fused DSP/PE (iteration 3) | *(pending)* |
| Requant DSP | 32 (2 × 16 lanes) | confirmed 2/lane at N=4 | *(pending)* |
| Total DSP | 288 | | *(pending)* |
| u_array LUT | ~2,600 | ~10 LUT/PE × 256, scales N² | *(pending)* |
| ws_quant_unit LUT | ~4,000 | array 2.6k + 4× requant + glue | *(pending)* |
| ws_quant_unit FF | ~2,000 | skew Σ(i+1)·9b ≈ 1.2k + array ~640 | *(pending)* |
| BRAM | 1 RAMB36-equiv (2×64×128 as RAMB18) | same packing as N=4 | *(pending)* |
| WNS | within ~1 ns of N=4 | per-PE paths N-independent; RISK: load_mode/w_capture/valid broadcast fanout to 256 PEs | *(pending)* |

If WNS slips well past 1 ns, the broadcast fanout is the first suspect —
the fix direction is registered fanout duplication (a control-tree
pipeline stage), which would add 1 cycle to LOAD and shift feed timing;
flag before implementing.

## Conventions

- Result: bottom of column j yields `sum_i w[i][j] * a[i]`, i.e.
  `y = W_loaded^T · a`. Hosts store weight tiles **transposed** to compute
  `y = W·a`.
- Weight tile in the buffer: word `i` (at `tile_idx*K + i`) = resident row
  `i`, lane `j` (bits `[8j+7:8j]`) = column `j`. One word = N packed
  int8s (32-bit at N=4, 128-bit at N=16); K = words/tile = N, power of 2.
- Activation word: lane `i` feeds array row `i`; one word = one vector.
- Column outputs are skewed (column j lags column 0 by j cycles) and exit
  continuously, gated by `sum_valid` — no drain. Consumers deskew.
