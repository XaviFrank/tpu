`timescale 1ns / 1ps
//====================================================================
// ws_perf_counters: WS category mapping over the existing (unchanged)
// perf_counters hardware. PURE OBSERVER, same as the OS instance.
//
//   WS register map (CONTRACT - mirrored in docs/perf_map.md; the
//   AXI-Lite PERF_SEL/PERF_DATA pair reads it):
//     0: total_cycles   every cycle since perf_clear was last released
//     1: load_cycles    cycles in W_LOAD   (= N+2 per tile)
//     2: stream_cycles  cycles in W_STREAM (= n_vecs+1 per tile)
//     3: flush_cycles   cycles in W_FLUSH  (= 2N+1 per tile)
//     4: unused         reads 0
//     5: idle_cycles    cycles in W_IDLE
//     6: unused         reads 0
//     7: tile_swaps     count of w_capture strobes (1 per accepted go)
//   total=0 / idle=5 / tile-count=7 keep the OS map's indices so host
//   code shares offsets. Saturating counters, registered read port,
//   out-of-range addresses read 0 (all inherited from perf_counters).
//====================================================================
module ws_perf_counters #(
    parameter int NCTR = 8,
    parameter int CW   = 32
)(
    input  logic          clk,
    input  logic          rst_n,
    input  logic          perf_clear,

    // Moore flags from ws_controller (never decoded enum values)
    input  logic          in_load,
    input  logic          in_stream,
    input  logic          in_flush,
    input  logic          in_idle,
    input  logic          tile_swap,     // 1-cycle strobe per tile load

    input  logic [3:0]    perf_addr,
    output logic [CW-1:0] perf_data      // valid 1 cycle after perf_addr
);

    logic const_zero;
    assign const_zero = 1'b0;

    perf_counters #(
        .NCTR(NCTR), .CW(CW)
    ) u_ctrs (
        .clk        (clk),
        .rst_n      (rst_n),
        .perf_clear (perf_clear),
        .in_fetch   (in_load),      // index 1: load
        .in_feed    (in_stream),    // index 2: stream
        .in_settle  (in_flush),     // index 3: flush
        .in_drain   (const_zero),   // index 4: unused
        .in_idle    (in_idle),      // index 5: idle
        .chunk_done (const_zero),   // index 6: unused
        .tile_done  (tile_swap),    // index 7: tile_swaps
        .perf_addr  (perf_addr),
        .perf_data  (perf_data)
    );

endmodule
