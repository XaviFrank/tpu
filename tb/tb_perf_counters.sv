`timescale 1ns / 1ps
//====================================================================
// tb_perf_counters: unit test of the passive counter block
//   CW=8 override so saturation (hold at 255, no wrap) is reachable.
//   P1: total_cycles counts every cycle after clear release
//   P2: flag windows land in the right map slots
//   P3: read-port latency is exactly 1 cycle (back-to-back addresses)
//   P4: out-of-range address reads 0
//   P5: saturation - count past 2^CW-1, counter holds at all-ones
//   P6: perf_clear zeroes everything
//====================================================================
module tb_perf_counters;

    localparam int NCTR = 8;
    localparam int CW   = 8;   // small on purpose: saturation reachable

    logic clk = 0;
    logic rst_n, perf_clear;
    logic in_fetch, in_feed, in_settle, in_drain, in_idle;
    logic chunk_done, tile_done;
    logic [3:0]    perf_addr;
    logic [CW-1:0] perf_data;

    int errors = 0;

    perf_counters #(
        .NCTR(NCTR), .CW(CW)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .perf_clear (perf_clear),
        .in_fetch   (in_fetch),
        .in_feed    (in_feed),
        .in_settle  (in_settle),
        .in_drain   (in_drain),
        .in_idle    (in_idle),
        .chunk_done (chunk_done),
        .tile_done  (tile_done),
        .perf_addr  (perf_addr),
        .perf_data  (perf_data)
    );

    always #5 clk = ~clk;

    //----------------------------------------------------------------
    // Read one counter. CONTRACT: call at a negedge; the address is
    // presented mid-cycle, the DUT registers the value at the next
    // posedge (pre-increment), and we sample it at the next negedge.
    // Each call consumes exactly one clock cycle.
    //----------------------------------------------------------------
    task automatic read_ctr(input int addr, output logic [CW-1:0] val);
        perf_addr = 4'(addr);
        @(negedge clk);
        val = perf_data;
    endtask

    task automatic check_ctr(input string name, input int addr,
                             input logic [CW-1:0] exp);
        logic [CW-1:0] v;
        read_ctr(addr, v);
        if (v !== exp) begin
            errors++;
            $error("[%s] ctr[%0d] = %0d, expected %0d", name, addr, v, exp);
        end
    endtask

    //----------------------------------------------------------------
    // Main
    //----------------------------------------------------------------
    initial begin
        logic [CW-1:0] v;

        rst_n = 0; perf_clear = 0;
        in_fetch = 0; in_feed = 0; in_settle = 0; in_drain = 0; in_idle = 0;
        chunk_done = 0; tile_done = 0;
        perf_addr = '0;
        repeat (2) @(negedge clk);
        rst_n = 1;

        //--- P1: total_cycles counts every cycle after clear release -
        @(negedge clk) perf_clear = 1;
        @(negedge clk) perf_clear = 0;
        repeat (10) @(negedge clk);            // 10 counted posedges
        check_ctr("P1 total", 0, 8'd10);       // pre-increment sample
        $display("PASS [P1 total_cycles] (%0d errors so far)", errors);

        //--- P2: flag windows -> map slots ----------------------------
        // exact posedge counts per flag; flags never overlap here
        in_fetch = 1;  repeat (5) @(negedge clk); in_fetch = 0;
        in_feed = 1;   repeat (4) @(negedge clk); in_feed = 0;
        in_settle = 1; repeat (8) @(negedge clk); in_settle = 0;
        in_drain = 1;  repeat (4) @(negedge clk); in_drain = 0;
        in_idle = 1;   repeat (3) @(negedge clk); in_idle = 0;
        chunk_done = 1; @(negedge clk); chunk_done = 0; @(negedge clk);
        chunk_done = 1; @(negedge clk); chunk_done = 0; @(negedge clk);
        tile_done = 1;  @(negedge clk); tile_done = 0;
        check_ctr("P2 fetch",  1, 8'd5);
        check_ctr("P2 feed",   2, 8'd4);
        check_ctr("P2 settle", 3, 8'd8);
        check_ctr("P2 drain",  4, 8'd4);
        check_ctr("P2 idle",   5, 8'd3);
        check_ctr("P2 chunks", 6, 8'd2);
        check_ctr("P2 tiles",  7, 8'd1);
        $display("PASS [P2 flag map] (%0d errors so far)", errors);

        //--- P3: 1-cycle read latency, back-to-back addresses ---------
        perf_addr = 4'd1;
        @(negedge clk);
        perf_addr = 4'd2;                       // change immediately
        if (perf_data !== 8'd5) begin           // still shows ctr[1]
            errors++;
            $error("[P3] expected ctr[1]=5 one cycle after addr, got %0d", perf_data);
        end
        @(negedge clk);
        if (perf_data !== 8'd4) begin           // now ctr[2]
            errors++;
            $error("[P3] expected ctr[2]=4 next cycle, got %0d", perf_data);
        end
        $display("PASS [P3 read latency] (%0d errors so far)", errors);

        //--- P4: out-of-range address reads 0 -------------------------
        check_ctr("P4 addr 12", 12, 8'd0);
        $display("PASS [P4 out-of-range] (%0d errors so far)", errors);

        //--- P5: saturation - hold at all-ones, no wrap ----------------
        in_drain = 1; repeat (300) @(negedge clk); in_drain = 0;
        check_ctr("P5 drain saturated", 4, 8'hFF);
        check_ctr("P5 total saturated", 0, 8'hFF);
        $display("PASS [P5 saturation] (%0d errors so far)", errors);

        //--- P6: clear zeroes everything ------------------------------
        @(negedge clk) perf_clear = 1;
        @(negedge clk) perf_clear = 0;
        check_ctr("P6 fetch cleared",  1, 8'd0);
        check_ctr("P6 feed cleared",   2, 8'd0);
        check_ctr("P6 settle cleared", 3, 8'd0);
        check_ctr("P6 drain cleared",  4, 8'd0);
        check_ctr("P6 idle cleared",   5, 8'd0);
        check_ctr("P6 chunks cleared", 6, 8'd0);
        check_ctr("P6 tiles cleared",  7, 8'd0);
        $display("PASS [P6 clear] (%0d errors so far)", errors);

        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
