`timescale 1ns / 1ps
//====================================================================
// tb_tile_buffer: contract test for the BRAM tile store
//   T1: write pattern everywhere, read back word by word
//   T2: read latency - rd_data reflects the PREVIOUS rd_en cycle's
//       address, and holds when rd_en is low
//   T3: back-to-back streaming reads, each 1 cycle behind its address
//   T4: write addr X while reading addr Y (Y != X) in the same cycle
//====================================================================
module tb_tile_buffer;

    localparam int DEPTH      = 64;
    localparam int WORD_WIDTH = 32;
    localparam int AW         = $clog2(DEPTH);

    logic                  clk = 0;
    logic                  wr_en;
    logic [AW-1:0]         wr_addr;
    logic [WORD_WIDTH-1:0] wr_data;
    logic                  rd_en;
    logic [AW-1:0]         rd_addr;
    logic [WORD_WIDTH-1:0] rd_data;

    int errors = 0;

    tile_buffer #(
        .DEPTH(DEPTH), .WORD_WIDTH(WORD_WIDTH)
    ) dut (
        .clk     (clk),
        .wr_en   (wr_en),
        .wr_addr (wr_addr),
        .wr_data (wr_data),
        .rd_en   (rd_en),
        .rd_addr (rd_addr),
        .rd_data (rd_data)
    );

    always #5 clk = ~clk;

    //----------------------------------------------------------------
    // Deterministic per-address pattern with 4 distinct bytes
    //----------------------------------------------------------------
    function automatic logic [WORD_WIDTH-1:0] pat(input int a);
        logic [7:0] a8;
        a8  = 8'(a);
        pat = {a8 ^ 8'hA5, a8 + 8'h33, ~a8, a8};
    endfunction

    task automatic check(input string name, input logic [WORD_WIDTH-1:0] exp);
        if (rd_data !== exp) begin
            errors++;
            $error("[%s] rd_data = %h, expected %h", name, rd_data, exp);
        end
    endtask

    //----------------------------------------------------------------
    // Main
    //----------------------------------------------------------------
    initial begin
        wr_en = 0; wr_addr = '0; wr_data = '0;
        rd_en = 0; rd_addr = '0;
        repeat (2) @(negedge clk);

        //--- T1: write pattern everywhere, read back word by word ----
        for (int a = 0; a < DEPTH; a++) begin
            @(negedge clk);
            wr_en   = 1;
            wr_addr = AW'(a);
            wr_data = pat(a);
        end
        @(negedge clk) wr_en = 0;

        for (int a = 0; a < DEPTH; a++) begin
            @(negedge clk);
            rd_en   = 1;
            rd_addr = AW'(a);
            @(negedge clk);
            rd_en = 0;
            check("T1 readback", pat(a));
        end
        $display("PASS [T1 write/readback] (%0d errors so far)", errors);

        //--- T2: latency contract + hold when rd_en low --------------
        @(negedge clk);
        rd_en   = 1;
        rd_addr = AW'(5);
        @(negedge clk);
        rd_en   = 0;
        rd_addr = AW'(9);            // address changes, but rd_en is low
        check("T2 latency", pat(5)); // reflects PREVIOUS rd_en cycle's addr
        @(negedge clk);
        check("T2 hold", pat(5));    // no rd_en: rd_data must hold
        @(negedge clk);
        check("T2 hold2", pat(5));
        $display("PASS [T2 latency+hold] (%0d errors so far)", errors);

        //--- T3: back-to-back streaming reads -------------------------
        @(negedge clk);
        rd_en   = 1;
        rd_addr = AW'(0);
        for (int a = 1; a < DEPTH; a++) begin
            @(negedge clk);
            check("T3 stream", pat(a - 1));  // 1 cycle behind its address
            rd_addr = AW'(a);
        end
        @(negedge clk);
        check("T3 stream last", pat(DEPTH - 1));
        rd_en = 0;
        $display("PASS [T3 streaming] (%0d errors so far)", errors);

        //--- T4: simultaneous write X / read Y (Y != X) ---------------
        @(negedge clk);
        wr_en   = 1;
        wr_addr = AW'(10);
        wr_data = 32'hDEAD_BEEF;
        rd_en   = 1;
        rd_addr = AW'(20);
        @(negedge clk);
        wr_en = 0;
        check("T4 read-during-write", pat(20));
        rd_addr = AW'(10);           // now confirm the write landed
        @(negedge clk);
        rd_en = 0;
        check("T4 write landed", 32'hDEAD_BEEF);
        $display("PASS [T4 write-while-read] (%0d errors so far)", errors);

        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
