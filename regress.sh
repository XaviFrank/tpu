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
                         rtl/axi_lite_ctrl.sv rtl/ws_tpu_top.sv tb/tb_axi_lite_ctrl.sv

[ $FAIL -eq 0 ] && echo "=== REGRESSION GREEN ===" || { echo "=== REGRESSION FAILED ==="; }
exit $FAIL
