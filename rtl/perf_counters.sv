`timescale 1ns / 1ps
//====================================================================
// perf_counters: passive instrumentation with an indexed read port.
//   PURE OBSERVER: no output feeds control logic anywhere; deleting
//   this module must not change any behavior.
//
//   Register map (CONTRACT - mirrored in docs/perf_map.md; Phase 3's
//   AXI-Lite/Python will read it):
//     0: total_cycles   every cycle since perf_clear was last released
//     1: fetch_cycles   cycles with in_fetch
//     2: feed_cycles    cycles with in_feed
//     3: settle_cycles  cycles with in_settle
//     4: drain_cycles   cycles with in_drain
//     5: idle_cycles    cycles with in_idle
//     6: chunks_completed   count of chunk_done strobes
//     7: tiles_completed    count of tile_done strobes
//
//   Counters SATURATE at all-ones (no wrap). perf_clear zeroes all
//   counters (that cycle does not count). Read port is registered:
//   perf_data is valid 1 cycle after perf_addr (same contract as
//   tile_buffer reads); out-of-range addresses read 0.
//====================================================================
module perf_counters #(
    parameter int NCTR = 8,
    parameter int CW   = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic perf_clear,           // strobe: zero all counters

    // state flags from the controller (explicit Moore flags, never
    // decoded enum values)
    input  logic in_fetch,
    input  logic in_feed,
    input  logic in_settle,
    input  logic in_drain,
    input  logic in_idle,
    input  logic chunk_done,           // 1-cycle strobe per chunk
    input  logic tile_done,            // 1-cycle strobe per tile

    // indexed read port (the future AXI register map)
    input  logic [3:0]    perf_addr,
    output logic [CW-1:0] perf_data    // valid 1 cycle after perf_addr
);

    logic [CW-1:0] ctr [NCTR];
    logic          inc [NCTR];

    assign inc[0] = 1'b1;              // total_cycles
    assign inc[1] = in_fetch;
    assign inc[2] = in_feed;
    assign inc[3] = in_settle;
    assign inc[4] = in_drain;
    assign inc[5] = in_idle;
    assign inc[6] = chunk_done;
    assign inc[7] = tile_done;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NCTR; i++)
                ctr[i] <= '0;
        end else if (perf_clear) begin
            for (int i = 0; i < NCTR; i++)
                ctr[i] <= '0;
        end else begin
            for (int i = 0; i < NCTR; i++)
                if (inc[i] && ctr[i] != '1)   // saturate, never wrap
                    ctr[i] <= ctr[i] + 1'b1;
        end
    end

    // registered read, 1-cycle latency; out-of-range reads 0
    always_ff @(posedge clk) begin
        if (perf_addr < NCTR) perf_data <= ctr[perf_addr];
        else                  perf_data <= '0;
    end

endmodule
