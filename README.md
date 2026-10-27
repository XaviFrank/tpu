# TPU-Inspired Transformer Accelerator on the Kria KV260

A from-scratch, TPU-style INT8 accelerator in SystemVerilog targeting the
AMD/Xilinx Kria KV260 (Zynq UltraScale+ MPSoC), built to run the
transformer blocks of **Qwen3-0.6B** — not just matrix multiplication,
but the normalization, activation, rotary-embedding, and attention
pieces around it.

The design is driven by measurement: every performance change was
predicted on paper first, then checked against on-chip cycle counters.

## Highlights

- **16×16 weight-stationary systolic array** with a tiling engine,
  per-tile requantization, and a 2-stage registered-product MAC.
- **Dual-clock design:** a fast compute clock (`core_clk`, 4.7 ns
  constraint) and a 100 MHz AXI-Lite control plane, joined by
  two-flop synchronizers and pulse-sync clock-domain crossings.
- **Beyond matmul:** RMSNorm, SwiGLU gating (SiLU LUT), RoPE, residual
  add, and an attention unit with a quantized, layer-addressed KV cache.
- **Bit-exact against real model weights:** a Qwen3-0.6B layer-0
  `q_proj` tile computes bit-exact against the Python golden model on
  the KV260 itself. Layer 0's attention and FFN projections are
  covered by the same pipeline in simulation.
- **Self-checking regression:** 38 testbenches under Icarus Verilog,
  run with one script (`./regress.sh`) on every change.
- **Cycle counters on chip:** saturating perf counters readable over
  AXI-Lite, with a documented register map.

## Architecture

```
            PS (ARM, PYNQ host)
                    │ AXI-Lite (100 MHz)
            ┌───────▼────────┐
            │  axi_lite_ctrl │  control / status / perf registers
            └───────┬────────┘
     CDC (sync_2ff, pulse_sync) ─┼───────────────────────────
                    │ core_clk
     ┌──────────────▼───────────────┐
     │      ws_tiling_engine        │  walks M/N/K tiles
     │  ┌──────────┐  ┌───────────┐ │
     │  │ws_control│→ │ ws_array  │ │  16×16 weight-stationary PEs
     │  └──────────┘  └─────┬─────┘ │
     │        tile_buffer   │       │
     │                ws_quant_unit │  requant INT32 → INT8
     └──────────────────────┬───────┘
                            ▼
   rmsnorm · swiglu_gate · rope · residual_add · attn_kv_top
```

| Block | Files |
|---|---|
| Weight-stationary array | `ws_pe.sv`, `ws_array.sv`, `ws_controller.sv`, `ws_unit.sv` |
| Tiling & top level | `ws_tiling_engine.sv`, `ws_tpu_top.sv`, `ws_tpu_tiled_top.sv` |
| Quantization | `quant_engine.sv`, `requant.sv`, `ws_quant_unit.sv`, `recip.sv` |
| Control plane & CDC | `axi_lite_ctrl.sv`, `sync_2ff.sv`, `pulse_sync.sv`, `bus_handshake.sv` |
| Transformer ops | `rmsnorm.sv`, `swiglu_gate.sv`, `rope.sv`, `q_scale.sv`, `residual_add.sv` |
| Attention & KV cache | `attn_kv_top.sv`, `attn_scan_gqa.sv`, `attn_finalize.sv`, `kv_*.sv` |
| Output-stationary baseline | `os_pe.sv`, `systolic_array.sv`, `controller.sv`, `tiling_engine.sv` |
| Instrumentation | `perf_counters.sv`, `ws_perf_counters.sv` |

An output-stationary array was built first and kept as a baseline; the
weight-stationary variant is the main design.

## Measured results

**Weight-stationary cycle budget**, verified at both N=4 and N=16:

| Phase | Cycles per tile |
|---|---|
| Weight load | N + 2 |
| Stream | n_vecs + 1 |
| Flush | 2N + 1 |
| Busy total | 3N + 4 + n_vecs |

These formulas are asserted in regression, so a pipeline change that
shifts timing fails the suite instead of passing silently.

**Double-buffering the output-stationary baseline** (16×16×16 problem,
64 chunks), predicted before measuring and matched exactly:

| Metric | Before | Double-buffered | Change |
|---|---|---|---|
| Total cycles | 801 | 609 | −24% |
| Fetch cycles | 320 | 128 | −60% |
| Utilization | 31.9% | 42.0% | +10.1 pts |

Full write-ups: [`docs/perf_baseline.md`](docs/perf_baseline.md),
[`docs/ws_design_notes.md`](docs/ws_design_notes.md),
[`docs/kv_cache_design_notes.md`](docs/kv_cache_design_notes.md),
[`docs/attention_read_design_notes.md`](docs/attention_read_design_notes.md),
and the perf register map in [`docs/perf_map.md`](docs/perf_map.md).

## Repository layout

```
rtl/          synthesizable SystemVerilog + LUT/ROM .memh files
tb/           self-checking testbenches
constraints/  Vivado timing constraints (.xdc)
software/     Qwen3-0.6B quantize/pack pipeline and board driver (Python)
docs/         design notes and measured performance
bitstreams/   prebuilt KV260 bitstream
regress.sh    full regression runner
```

## Running the regression

Requires [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build)
(Icarus Verilog) and bash.

```bash
OSS_CAD_SUITE=/path/to/oss-cad-suite ./regress.sh
```

Each testbench prints `ALL TESTS PASSED` on success; the script exits 0
only if every one passes. Logs land in `build/`.

## Model pipeline

`software/` quantizes real Qwen3-0.6B weights to INT8, packs them into
the buffer layout the hardware reads, and produces INT32 golden outputs
to compare against. It needs only `numpy` and `tokenizers` (no torch).
See [`software/README.md`](software/README.md).

## Status

Working: the weight-stationary core, AXI-Lite control, requantization,
and the transformer blocks above, each verified in simulation; on-board
bit-exact execution of a real Qwen3 projection tile.

Next: chaining the blocks into a full decoder layer on the board, and
DMA-driven weight delivery (`software/dma_weight_load.py`) for
end-to-end token generation.

## Author

Xavier Frank — Electrical and Computer Engineering, Princeton University
