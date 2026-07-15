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

[ $FAIL -eq 0 ] && echo "=== REGRESSION GREEN ===" || { echo "=== REGRESSION FAILED ==="; }
exit $FAIL
