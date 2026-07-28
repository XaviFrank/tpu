# Design: poke/peek registers — real hardware smoke test

**Date:** 2026-07-28
**Source spec:** user-provided "Spec: poke/peek registers — real hardware smoke test"
**Status:** approved, ready for implementation planning

## Purpose

The minimal register-driven path to load a complete weight tile and
activation vectors onto real silicon, trigger a run, and read back a
complete, checkable result — entirely over AXI-Lite. Deliberately slow
(one 32-bit word per register write). This is **not** the real
data-movement path; DMA/AXI-HP remains separate, deferred work. The
only job here is proving `ws_tpu_top` computes a real, correct answer
on hardware before Block Design work continues.

## Findings that changed the source spec

Three premises in the source spec did not survive contact with the RTL.
All three were confirmed by reading current code, not assumed.

### F1. There is no result buffer (scope change)

The source spec's §1 (`PEEK_ADDR` buf_sel "result=2") and §3 step 5
("peek the result buffer, full depth") assume result storage exists. It
does not. `ws_quant_unit` emits `q8_out[N]`/`q8_valid[N]` as a live,
column-skewed **streaming tap** with zero storage — its header calls it
"a true streaming tap" — and `ws_tpu_top` forwards those straight to
top-level ports. Nothing stores results anywhere in the design. This is
also why the routed build reports 179 unloaded `q8_out[*]` nets
(RTSTAT-10).

Consequence: the source spec's §5 claim that this is "a register-map
addition to existing control-plane logic, not a new module" does not
hold. A new capture module is required.

### F2. 32-bit AXI-Lite vs 128-bit buffer words

At the synth parameterization (N=16) one buffer word is
`N*DATA_WIDTH` = **128 bits** (`tile_buffer.WORD_WIDTH`), but
`POKE_DATA` is a single 32-bit register. The source spec's
one-write-one-word framing is not physically realizable. Poke requires
four lane writes plus a commit; peek requires four reads.

### F3. The §2 CDC concern is already solved by an existing primitive

`bus_handshake`'s `dst_valid` is already documented as "1-cycle pulse
per delivery" (`rtl/bus_handshake.sv:31`). Gating a buffer write enable
on `dst_valid` yields exactly-once semantics with no new mechanism —
precisely the level-sensitive firing hazard §2 warned about, already
avoided by the primitive in use.

### F4. §6 step ordering is internally inconsistent

Source §6 step 1 requires a TB covering "write + peek-back" for the
weight buffer, but defers `PEEK_ADDR`/`PEEK_DATA` to step 3. Step 1's
own test could not run. **Resolved:** peek for the weight/activation
buffers lands in step 1, because it is what makes step 1 testable; step
3 then only *extends* peek to the result banks. This matches source §1's
own reasoning that peek exists so "a poke can be verified by reading it
back before ever triggering `go`."

## Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Capture **both** the INT32 `sum_out` and INT8 `q8` streams | Isolates a requant bug from an array bug on hardware |
| D2 | Gate the capture buffer behind `DEBUG_CAPTURE`, default 1 | Keeps observability; leaves a way to reclaim area later |
| D3 | `ws_tpu_top` only; `ws_tpu_tiled_top` ties the new ports inert | Matches the stated purpose; minimises added core-domain fanout |
| D4 | Address everything as a flat array of 32-bit elements | One scheme covers all four address spaces |
| D5 | Poke sends address+data as **one** handshake payload | Makes the pair atomic; deletes an ordering hazard (see §3) |
| D6 | Peek reads the **real** buffers via a mux in `ws_unit` | A shadow copy costs more area and proves less (see §4) |
| D7 | Peek read path is registered on both sides | Keeps a wide mux off the critical path (see §5) |
| D8 | New `tb/tb_poke_peek.sv` at N=K=16 | Only size that exercises the 128-bit word path |

## 1. Architecture

Five pieces, one of them a new module:

- `rtl/axi_lite_ctrl.sv` — four new registers, one new STATUS bit, new
  `bus_handshake` instances.
- `rtl/ws_result_capture.sv` (**new**) — deskews and stores both output
  streams.
- `rtl/ws_unit.sv` — debug read mux in front of the two `tile_buffer`
  read ports.
- `rtl/ws_quant_unit.sv` — pass-through of the debug ports.
- `rtl/ws_tpu_top.sv` — instantiates the capture buffer, wires
  poke/peek, carries the new parameters.

`ws_tpu_top` gains two parameters it does not currently have:
`DEBUG_CAPTURE` (default `1`) and `MAX_N_VECS` (default `16`). The
latter sizes the capture banks; `VEC_W = 4` bounds `n_vecs` at 15, so 16
covers the full range. Both are forwarded to `ws_result_capture`, and
`DEBUG_CAPTURE` also reaches `ws_quant_unit`/`ws_unit` to gate the debug
read mux. The existing register-map packing guards in `ws_tpu_top` get a
companion guard: `MAX_N_VECS` must be at least `2**VEC_W - 1`, or a run
could stream more vectors than the banks can hold.

`ws_tpu_tiled_top` connects the new `axi_lite_ctrl` ports inert, with a
header note that the smoke path lives on `ws_tpu_top`.

## 2. Register map

Offsets confirmed against the real decode: `A_KTILES = 4'h6` (`0x18`),
`A_NTILES = 4'h7` (`0x1C`), so the next four word slots are free.
`s_axi_awaddr[5:0]` decodes on `addr[5:2]`, giving 16 words — room for
all four.

| offset | decode | name | dir | fields |
|---|---|---|---|---|
| `0x20` | `4'h8` | `POKE_ADDR` | W | `[5:0]` word addr, `[7:6]` lane, `[8]` buf_sel (0=weight, 1=activation) |
| `0x24` | `4'h9` | `POKE_DATA` | W | 32-bit lane value; a write with `lane == 3` **commits** the assembled 128-bit word |
| `0x28` | `4'hA` | `PEEK_ADDR` | W | `[5:0]` idx0, `[9:6]` idx1, `[11:10]` buf_sel |
| `0x2C` | `4'hB` | `PEEK_DATA` | R | selected 32-bit element |

`PEEK_ADDR.buf_sel`: `0` = weight buffer, `1` = activation buffer,
`2` = result q8, `3` = result INT32.

Flat 32-bit element addressing (D4):

| buf_sel | idx0 | idx1 | element returned |
|---|---|---|---|
| 0 weight | word 0..63 | lane `[1:0]` | 32 bits of the 128-bit word |
| 1 activation | word 0..63 | lane `[1:0]` | 32 bits of the 128-bit word |
| 2 result q8 | vector 0..15 | lane `[1:0]` | 4 packed int8 (columns `lane*4 .. lane*4+3`) |
| 3 result INT32 | vector 0..15 | column `[3:0]` | one INT32 accumulator value |

Field widths are taken from real parameters, not guessed:
`tile_buffer DEPTH = 64` → 6-bit word address covering **all** 64 words,
which reaches a complete K=16 weight tile and all 16 activation vectors;
`N*DATA_WIDTH = 128` → 2-bit lane.

New status bit: **`STATUS[3] = poke_busy`** (see §3). Existing bits
unchanged: `[0]` busy, `[1]` done, `[2]` cfg_err.

`POKE_ADDR`/`PEEK_ADDR` support AXI-side readback, matching the existing
`N_VECS`/`PERF_SEL` convention.

## 3. Poke path

**Deviation from source §2, deliberate.** Source §2 says treat
`POKE_ADDR` like `N_VECS` via `bus_handshake`. This design does not.
`POKE_ADDR` stays a plain AXI-domain register with no CDC, and address
and data cross together as a **single payload**.

Mechanism:

1. `POKE_ADDR` writes land in an AXI-domain register. No crossing.
2. A `POKE_DATA` write deposits 32 bits into `poke_stage[lane]`, a
   128-bit AXI-domain staging register.
3. On a write with `lane == 3`, the AXI side forms the payload
   `{buf_sel, word_addr, poke_stage}` — 1 + 6 + 128 = **135 bits** — and
   pulses one `bus_handshake #(.W(135))`.
4. Core side: `dst_valid` is a 1-cycle pulse (F3), so it drives
   `wr_en_w` or `wr_en_a` directly for exactly one core cycle, with
   `dst_data` supplying address and data.

Why this beats two independent handshakes: it makes address and data
**atomic**. With separate crossings, a host that writes `POKE_ADDR` and
then `POKE_DATA` too quickly could commit data against a stale address
and silently corrupt the buffer — a settle-before-poke ordering hazard
with no natural backstop. Folding them removes the hazard by
construction and uses *fewer* handshake instances. Exactly-once is
inherited from `dst_valid` rather than built, which is what source §2
asked for.

**Hazard the source spec did not cover.** `bus_handshake` ignores
`src_send` while `src_busy`, so a commit issued while the previous one
is still in flight would be **silently dropped** — a lost word, i.e. a
correctness failure, not a slow path. Mitigation: `STATUS[3] =
poke_busy` exposes `src_busy`; the host polls it before each commit
write. The TB does this, and it is the contract PYNQ inherits.

Duplicate delivery is harmless if it ever occurs: the same (address,
data) pair rewritten to the same location is idempotent. Only drops
matter, and `poke_busy` covers drops.

Poke covers weight and activation only. Result banks are chip-written;
poke has no path to them.

## 4. Peek path

Reuses the proven `PERF_SEL`/`PERF_DATA` round trip unchanged in shape:
`PEEK_ADDR` crosses to the core via `bus_handshake`, the core performs a
**registered** read of the selected element, and the 32-bit result
crosses back via a second `bus_handshake` into `PEEK_DATA`. Host
contract is identical to PERF and reuses its documented budget: write
`PEEK_ADDR`, wait ~20 aclk, read `PEEK_DATA`.

**Reading the weight/activation buffers (D6).** `tile_buffer`'s header
forbids a second read port — "BRAM inference is fragile - keep exactly
this read idiom: no reset on rd_data/mem, no combinational read, no 2nd
read port." So the debug read does **not** add a port. Instead a mux
sits in front of the existing read port inside `ws_unit`, where
`rd_en_w`/`rd_addr_w`/`rd_data_w` are already plain internal wires
between `ws_controller` and `tile_buffer`. The controller wins
unconditionally while busy.

Consequences:

- `tile_buffer` is untouched and still sees exactly one read port, so
  BRAM inference is preserved.
- `ws_controller` is untouched.
- A peek issued mid-run returns garbage but **cannot** disturb compute.
  Documented contract: peek while idle. The smoke test peeks before `go`
  and after `done`.
- `rd_data` is registered one cycle after `rd_en`, so the debug read
  captures it into a holding register before shipping it back.

**Alternative considered and rejected:** shadowing poked data in the
control plane so `ws_unit` need not be touched. Worse on both axes — it
needs 64 words x 128 bits x 2 buffers = **16 Kb** of shadow storage,
more than the capture buffer itself, and it would only prove the AXI
path worked, not that `tile_buffer` holds the right bits. That defeats
the entire purpose of source §3 step 3, which exists to separate
"did poke work" from "did compute work".

## 5. Result capture (`ws_result_capture`, gated by `DEBUG_CAPTURE`)

Reuses `ws_tiling_engine`'s proven accumulate-on-write bank shape: one
bank per column, each with its own `v_cnt[j]` write index, capturing
each column's value as the skewed stream arrives. Per-column write
ports are what keep this in fabric/LUTRAM rather than BRAM — the same
outcome the tiling engine's 8 Kb banks already demonstrate.

Storage at N=16, `MAX_N_VECS=16`:

| bank | size |
|---|---|
| q8 | 16 x 16 x 8b = 2,048 b |
| INT32 sum | 16 x 16 x 32b = 8,192 b |
| **total** | **10,240 b (~10 Kb), 0 DSP** |

Plus per-column write indices, 16 x 4b = 64 flops.

Reset style follows the house rule: `v_cnt` is control (async reset),
the banks themselves are datapath (no reset) — the write-before-read
discipline covers post-reset X, exactly as the engine's banks do.

**Capture arming — this is what makes §6 step 6 work.** Each column's
`v_cnt[j]` is cleared by `core_go`, exactly as `ws_tiling_engine` clears
its own on the inner go. So every run rewrites the banks from vector 0,
and a second run cannot leave stale values from the first at any index
it writes. Capture is otherwise free-running on `q8_valid[j]` /
`sum_valid[j]`; there is no separate arm/disarm control to get wrong.
Indices *beyond* the current run's `n_vecs` retain the previous run's
data by design — the TB reads only `0 .. n_vecs-1`, and round two of the
smoke test uses the same `n_vecs`, so it overwrites every index it
checks. This is a deliberate property, not an accident: it is what
proves poke/peek overwrites rather than only working once.

**Timing risk and its mitigation (D7).** The dominant cost is not the
storage but the **peek read mux**: selecting 1 of 256 INT32 elements is
wide, and it lands in the core clock domain that already owns the
critical path (measured 97% route delay at 0 logic levels). The read is
therefore **registered on both sides** — address register, then bank
read, then output register — never combinational. Peek has a 20-aclk
budget, so spending 2-3 core cycles costs nothing and keeps the mux off
the critical path entirely.

`DEBUG_CAPTURE = 0` removes the banks, the read mux, and the result
peek path together. `DEBUG_CAPTURE = 1` is the default and the smoke
build.

## 6. Testbench

`tb/tb_poke_peek.sv` (new), at **N = K = 16** — the synth
parameterization, and the only size that exercises the 128-bit word path
(D8). Driven **entirely through the AXI-Lite interface**: no whitebox
writes to `wr_en_w`/`wr_data_w`, no hierarchical pokes. It is literally
the sequence PYNQ will run.

Sequence, per source §3:

1. Poke a known weight tile, **full depth** (all 16 words of the tile).
2. Poke known activation vectors.
3. **Peek both back before `go`** and compare byte for byte. This is the
   step that makes a later failure unambiguous about which layer broke.
4. Set `N_VECS`, write `CTRL.go`, poll `STATUS.done`.
5. Peek the result banks, full depth — **both** the q8 values and the
   INT32 sums — against a hand-computable golden.
6. Repeat with a second, different tile+vector pair to prove no stale
   state and that poke overwrites correctly rather than only working
   once.

Golden values are chosen small and structured so results are
hand-verifiable, with `scale_mult = 1`, `shift_amt = 0`, `relu` off, so
q8 is simply the clamped INT32 sum and the two readbacks cross-check
each other.

TB conventions follow the house rules: drive negedge / sample posedge,
`!==` comparisons, self-checking with an `ALL TESTS PASSED` banner.
`poke_busy` is polled before every commit.

## 7. Commit steps

Each step must reach a green `./regress.sh` before the next begins.

1. `POKE_ADDR`/`POKE_DATA` and `PEEK_ADDR`/`PEEK_DATA` for the
   **weight** buffer, including the `ws_unit` debug read mux and
   `poke_busy`. TB: poke a full tile, peek it back, compare. (Peek is in
   step 1 per F4, not step 3.)
2. Extend poke and peek to the **activation** buffer.
3. `ws_result_capture` + result peek for both q8 and INT32 + the full
   §6 sequence including the second round.
4. Add `tb_poke_peek` to `regress.sh`; full regression green at 18 TBs.

## 8. Out of scope

- Real bulk data movement (DMA/AXI-HP). This path is intentionally slow
  and is not meant to survive into the real-throughput design.
- Multi-tile sweeps via poke/peek. One tile proves the compute pipeline;
  tiling-level smoke testing waits for DMA, since poking multiple full
  tiles by hand is not a reasonable use of this mechanism.
- Poking the result banks. Chip-written only.

## 9. Vivado handoff gate (open until measured)

No DSP inference is expected anywhere in this addition — pure
register/buffer-write logic and fabric banks. A DSP appearing in
`ws_result_capture` or the control plane is a bug signal.

**Hold slack must be re-measured on the combined build before this work
is called done.** Hold margin has shrunk at every integration step so
far (0.107 → 0.058 → 0.055 ns), and this addition puts new fanout in the
same core domain that owns the critical path. The pre-integration
baseline to compare against is the routed `ws_tpu_top` run of
2026-07-16: **WNS +0.559 ns, WHS +0.055 ns**, 0 failing endpoints.

This gate cannot be closed from simulation. It stays open until the
human reports the number from a Vivado run of the combined build.
