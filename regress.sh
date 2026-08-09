#!/bin/bash
# regress.sh - full TB regression via Icarus. Exit 0 = all green.
# Self-activating: carries its own toolchain PATH so no shell profile
# or global Windows PATH entry is required (keeps Vivado's env clean).
export PATH="/c/dev/oss-cad-suite/bin:/c/dev/oss-cad-suite/lib:$PATH"

set -u
FLAGS="-g2012"
BUILD=build
mkdir -p $BUILD
FAIL=0

run_tb () {
    local name=$1; shift
    echo "=== $name ==="
    if ! iverilog $FLAGS -o $BUILD/$name.vvp "$@" 2> $BUILD/$name.compile.log; then
        echo "COMPILE FAILED: $name"; cat $BUILD/$name.compile.log; FAIL=1; return
    fi
    vvp $BUILD/$name.vvp > $BUILD/$name.log 2>&1
    if grep -q "ALL TESTS PASSED" $BUILD/$name.log; then
        echo "PASS: $name"
    else
        echo "FAIL: $name"; grep -E "Error|ERROR" $BUILD/$name.log | head -20; FAIL=1
    fi
}

run_tb tb_pe_os          rtl/mac_unit.sv rtl/os_pe.sv tb/tb_pe_os.sv
run_tb tb_tile_buffer    rtl/tile_buffer.sv tb/tb_tile_buffer.sv
run_tb tb_perf_counters  rtl/perf_counters.sv tb/tb_perf_counters.sv
run_tb tb_systolic_array rtl/mac_unit.sv rtl/os_pe.sv rtl/systolic_array.sv tb/tb_systolic_array.sv
run_tb tb_matmul_unit    rtl/mac_unit.sv rtl/os_pe.sv rtl/systolic_array.sv \
                         rtl/input_skew.sv rtl/tile_buffer.sv rtl/controller.sv \
                         rtl/matmul_unit.sv tb/tb_matmul_unit.sv
run_tb tb_tiling_engine  rtl/mac_unit.sv rtl/os_pe.sv rtl/systolic_array.sv \
                         rtl/input_skew.sv rtl/tile_buffer.sv rtl/controller.sv \
                         rtl/matmul_unit.sv rtl/perf_counters.sv rtl/tiling_engine.sv \
                         tb/tb_tiling_engine.sv

# --- WS (weight-stationary) variant - parallel build, OS stays intact ---
run_tb tb_ws_pe          rtl/ws_pe.sv tb/tb_ws_pe.sv
run_tb tb_ws_array       rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         tb/tb_ws_array.sv
run_tb tb_ws_controller  rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         tb/tb_ws_controller.sv

# --- requant/ReLU (shared INT32->INT8 core; vectors from gen_requant_vectors.py) ---
run_tb tb_requant        rtl/requant.sv tb/tb_requant.sv
run_tb tb_quant_engine   rtl/mac_unit.sv rtl/os_pe.sv rtl/systolic_array.sv \
                         rtl/input_skew.sv rtl/tile_buffer.sv rtl/controller.sv \
                         rtl/matmul_unit.sv rtl/perf_counters.sv rtl/tiling_engine.sv \
                         rtl/requant.sv rtl/quant_engine.sv tb/tb_quant_engine.sv
run_tb tb_ws_quant_unit  rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         rtl/requant.sv rtl/ws_quant_unit.sv tb/tb_ws_quant_unit.sv

# --- N=16 scaling checkpoint (same WS RTL, larger parameterization) ---
run_tb tb_ws_n16         rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         rtl/requant.sv rtl/ws_quant_unit.sv tb/tb_ws_n16.sv

# --- AXI-Lite control wrapper (control plane; CDC from spec step 3 on) ---
run_tb tb_axi_lite_ctrl  rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         rtl/requant.sv rtl/ws_quant_unit.sv \
                         rtl/sync_2ff.sv rtl/pulse_sync.sv rtl/bus_handshake.sv \
                         rtl/axi_lite_ctrl.sv rtl/ws_result_capture.sv rtl/ws_tpu_top.sv tb/tb_axi_lite_ctrl.sv

# --- N=16 chip-level checkpoint (ws_tpu_top at the synth parameterization) ---
run_tb tb_ws_tpu_n16     rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         rtl/requant.sv rtl/ws_quant_unit.sv \
                         rtl/sync_2ff.sv rtl/pulse_sync.sv rtl/bus_handshake.sv \
                         rtl/axi_lite_ctrl.sv rtl/ws_result_capture.sv rtl/ws_tpu_top.sv tb/tb_ws_tpu_n16.sv

# --- WS tiling engine (multi-tile sweep over ws_quant_unit) ---
run_tb tb_ws_tiling_engine rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         rtl/requant.sv rtl/ws_quant_unit.sv \
                         rtl/ws_tiling_engine.sv tb/tb_ws_tiling_engine.sv

run_tb tb_ws_tpu_tiled_top rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         rtl/requant.sv rtl/ws_quant_unit.sv \
                         rtl/sync_2ff.sv rtl/pulse_sync.sv rtl/bus_handshake.sv \
                         rtl/axi_lite_ctrl.sv rtl/ws_tiling_engine.sv \
                         rtl/ws_tpu_tiled_top.sv tb/tb_ws_tpu_tiled_top.sv

# --- poke/peek hardware smoke path (AXI-Lite only, N=16) ---
run_tb tb_poke_peek      rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         rtl/requant.sv rtl/ws_quant_unit.sv \
                         rtl/sync_2ff.sv rtl/pulse_sync.sv rtl/bus_handshake.sv \
                         rtl/axi_lite_ctrl.sv rtl/ws_result_capture.sv rtl/ws_tpu_top.sv tb/tb_poke_peek.sv

# --- AXI4-Stream -> buffer adapter (DMA MM2S landing path) ---
run_tb tb_axi_stream_to_buf_adapter rtl/tile_buffer.sv \
                         rtl/axi_stream_to_buf_adapter.sv \
                         tb/tb_axi_stream_to_buf_adapter.sv

# --- DMA delivery path: adapter -> ws_tpu_top, buf_ready = !core_busy ---
run_tb tb_dma_weight_load rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         rtl/requant.sv rtl/ws_quant_unit.sv \
                         rtl/sync_2ff.sv rtl/pulse_sync.sv rtl/bus_handshake.sv \
                         rtl/axi_lite_ctrl.sv rtl/ws_result_capture.sv \
                         rtl/ws_tpu_top.sv rtl/axi_stream_to_buf_adapter.sv \
                         tb/tb_dma_weight_load.sv

# --- RMSNorm (first non-matmul unit) ---
run_tb tb_rmsnorm        rtl/requant.sv rtl/rmsnorm.sv tb/tb_rmsnorm.sv

# --- SwiGLU gate (SiLU + elementwise multiply + requant) ---
run_tb tb_swiglu_gate    rtl/requant.sv rtl/swiglu_gate.sv tb/tb_swiglu_gate.sv

# --- INT16 residual stream: rescale a stage output and add it back ---
run_tb tb_residual_add   rtl/residual_add.sv tb/tb_residual_add.sv

# --- RoPE phase generation (accumulate + seek + cos/sin) ---
run_tb tb_rope           rtl/rope.sv tb/tb_rope.sv

# --- Q scaling: RoPE's INT8 -> attn_kv_top's pre-scaled 32-bit query ---
run_tb tb_q_scale        rtl/q_scale.sv tb/tb_q_scale.sv

# --- attention output finalization: 1/l as mantissa + shift ---
run_tb tb_recip          rtl/recip.sv tb/tb_recip.sv

# --- attention output finalization: 1/l -> inverse rotation -> requant ---
run_tb tb_attn_finalize  rtl/recip.sv rtl/kv_irotate.sv rtl/requant.sv \
                         rtl/attn_finalize.sv tb/tb_attn_finalize.sv

# --- KV cache write path: rotation stage, both constructions ---
run_tb tb_kv_rotate      rtl/kv_rotate.sv tb/tb_kv_rotate.sv
run_tb tb_kv_quantize    rtl/kv_rotate.sv rtl/kv_quantize.sv tb/tb_kv_quantize.sv
run_tb tb_kv_cache_store rtl/kv_rotate.sv rtl/kv_quantize.sv \
                         rtl/kv_cache_store.sv tb/tb_kv_cache_store.sv
run_tb tb_kv_dequant     rtl/kv_rotate.sv rtl/kv_quantize.sv \
                         rtl/kv_cache_store.sv rtl/kv_unpack.sv \
                         rtl/kv_irotate.sv rtl/kv_dequant.sv tb/tb_kv_dequant.sv

# --- KV write path: INT8 K line (K moved off low-bit after the
# --- end-to-end study; V stays 3-bit) ---
run_tb tb_kv_pack_int8   rtl/kv_pack_int8.sv rtl/kv_unpack.sv                          tb/tb_kv_pack_int8.sv

# --- attention read path: streaming scan + online softmax (1 Q head) ---
run_tb tb_attn_scan      rtl/kv_unpack.sv rtl/attn_scan.sv tb/tb_attn_scan.sv
run_tb tb_attn_scan_gqa  rtl/kv_unpack.sv rtl/attn_scan.sv rtl/attn_scan_gqa.sv                          tb/tb_attn_scan_gqa.sv

# --- KV write path + attention read path, assembled end to end ---
run_tb tb_attn_e2e       rtl/kv_pack_int8.sv rtl/kv_rotate.sv                          rtl/kv_quantize.sv rtl/kv_cache_store.sv                          rtl/kv_unpack.sv rtl/attn_scan.sv                          rtl/attn_scan_gqa.sv tb/tb_attn_e2e.sv

# --- the whole attention unit as one synthesizable top, repeated go ---
run_tb tb_attn_kv_top    rtl/kv_pack_int8.sv rtl/kv_rotate.sv \
                         rtl/kv_quantize.sv rtl/kv_cache_store.sv \
                         rtl/kv_unpack.sv rtl/attn_scan.sv \
                         rtl/attn_scan_gqa.sv rtl/attn_kv_top.sv \
                         tb/tb_attn_kv_top.sv

# --- layer-offset addressing: two layers, one shared cache ---
run_tb tb_attn_kv_layers rtl/kv_pack_int8.sv rtl/kv_rotate.sv \
                         rtl/kv_quantize.sv rtl/kv_cache_store.sv \
                         rtl/kv_unpack.sv rtl/attn_scan.sv \
                         rtl/attn_scan_gqa.sv rtl/attn_kv_top.sv \
                         tb/tb_attn_kv_layers.sv

# --- build guard: the DEBUG_CAPTURE=0 (area reclaimed) configuration ---
run_tb tb_debug_capture_off rtl/ws_pe.sv rtl/ws_array.sv rtl/input_skew.sv \
                         rtl/tile_buffer.sv rtl/ws_controller.sv rtl/ws_unit.sv \
                         rtl/perf_counters.sv rtl/ws_perf_counters.sv \
                         rtl/requant.sv rtl/ws_quant_unit.sv \
                         rtl/sync_2ff.sv rtl/pulse_sync.sv rtl/bus_handshake.sv \
                         rtl/axi_lite_ctrl.sv rtl/ws_result_capture.sv \
                         rtl/ws_tpu_top.sv tb/tb_debug_capture_off.sv

[ $FAIL -eq 0 ] && echo "=== REGRESSION GREEN ===" || { echo "=== REGRESSION FAILED ==="; }
exit $FAIL
