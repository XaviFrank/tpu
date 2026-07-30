`timescale 1ns / 1ps
//====================================================================
// tb_axi_stream_to_buf_adapter: AXI4-Stream in, real tile_buffer out.
//
// Each adapter drives an ACTUAL tile_buffer and every check reads the
// data back through that buffer's read port, so a pass means the bytes
// genuinely landed in memory - not merely that wr_en/wr_addr/wr_data
// wiggled correctly. Two instances (weight-like and activation-like)
// prove the one-module-two-instantiations pattern; the module is
// target-agnostic, so they are identical by construction.
//
//   T1: 16-beat transfer with tlast, landing sequentially at 0..15
//   T2: tlast reset the counter - a second transfer starts at 0 again
//       and overwrites, rather than continuing from 16
//   T3: BACKPRESSURE - buf_ready held low mid-stream with tvalid high:
//       tready falls, no write happens, and every remaining beat still
//       lands in the right place. Data is checked, not just the count,
//       so a dropped OR duplicated beat both fail.
//   T4: wrap - a transfer longer than DEPTH wraps to 0 and sets the
//       sticky `wrapped` flag rather than corrupting silently
//   T5: the activation-side instance does all of T1 independently, and
//       the two instances do not disturb each other
//====================================================================
module tb_axi_stream_to_buf_adapter;

    localparam int WORD_W = 128;              // N*DATA_WIDTH at N=16
    localparam int DEPTH  = 64;
    localparam int AW     = 6;

    logic clk = 1'b0;
    logic rst_n;

    always #5 clk = ~clk;

    initial begin
        #2_000_000;
        $error("[watchdog] TIMEOUT");
        $finish;
    end

    int errors = 0;

    //----------------------------------------------------------------
    // Weight-side stream + adapter + buffer
    //----------------------------------------------------------------
    logic              w_tvalid, w_tready, w_tlast, w_buf_ready, w_wrapped;
    logic [WORD_W-1:0] w_tdata;
    logic              w_wr_en;
    logic [AW-1:0]     w_wr_addr;
    logic [WORD_W-1:0] w_wr_data;
    logic              w_rd_en;
    logic [AW-1:0]     w_rd_addr;
    logic [WORD_W-1:0] w_rd_data;

    axi_stream_to_buf_adapter #(
        .WORD_W(WORD_W), .DEPTH(DEPTH)
    ) u_adapt_w (
        .clk       (clk),
        .rst_n     (rst_n),
        .s_tvalid  (w_tvalid),
        .s_tready  (w_tready),
        .s_tdata   (w_tdata),
        .s_tlast   (w_tlast),
        .buf_ready (w_buf_ready),
        .wr_en     (w_wr_en),
        .wr_addr   (w_wr_addr),
        .wr_data   (w_wr_data),
        .wrapped   (w_wrapped)
    );

    tile_buffer #(
        .DEPTH(DEPTH), .WORD_WIDTH(WORD_W)
    ) u_buf_w (
        .clk     (clk),
        .wr_en   (w_wr_en),
        .wr_addr (w_wr_addr),
        .wr_data (w_wr_data),
        .rd_en   (w_rd_en),
        .rd_addr (w_rd_addr),
        .rd_data (w_rd_data)
    );

    //----------------------------------------------------------------
    // Activation-side stream + adapter + buffer (second instantiation)
    //----------------------------------------------------------------
    logic              a_tvalid, a_tready, a_tlast, a_buf_ready, a_wrapped;
    logic [WORD_W-1:0] a_tdata;
    logic              a_wr_en;
    logic [AW-1:0]     a_wr_addr;
    logic [WORD_W-1:0] a_wr_data;
    logic              a_rd_en;
    logic [AW-1:0]     a_rd_addr;
    logic [WORD_W-1:0] a_rd_data;

    axi_stream_to_buf_adapter #(
        .WORD_W(WORD_W), .DEPTH(DEPTH)
    ) u_adapt_a (
        .clk       (clk),
        .rst_n     (rst_n),
        .s_tvalid  (a_tvalid),
        .s_tready  (a_tready),
        .s_tdata   (a_tdata),
        .s_tlast   (a_tlast),
        .buf_ready (a_buf_ready),
        .wr_en     (a_wr_en),
        .wr_addr   (a_wr_addr),
        .wr_data   (a_wr_data),
        .wrapped   (a_wrapped)
    );

    tile_buffer #(
        .DEPTH(DEPTH), .WORD_WIDTH(WORD_W)
    ) u_buf_a (
        .clk     (clk),
        .wr_en   (a_wr_en),
        .wr_addr (a_wr_addr),
        .wr_data (a_wr_data),
        .rd_en   (a_rd_en),
        .rd_addr (a_rd_addr),
        .rd_data (a_rd_data)
    );

    //----------------------------------------------------------------
    // Independent write monitors: record (addr,data) for every write
    // the adapter actually issues. This is what makes "no beat dropped
    // or duplicated" checkable - the buffer alone cannot show a
    // duplicate that overwrote the same address with the same value.
    //----------------------------------------------------------------
    int w_writes = 0;
    int a_writes = 0;
    int w_seen_addr [256];

    always @(posedge clk) begin
        if (rst_n) begin
            if (w_wr_en === 1'b1) begin
                if (w_writes < 256) w_seen_addr[w_writes] = w_wr_addr;
                w_writes = w_writes + 1;
            end
            if (a_wr_en === 1'b1) a_writes = a_writes + 1;
        end
    end

    //----------------------------------------------------------------
    // Stream driver. Distinct per-beat payload so a mis-ordered or
    // duplicated beat cannot alias into a pass.
    //----------------------------------------------------------------
    function automatic [WORD_W-1:0] payload(input int seed, input int i);
        logic [WORD_W-1:0] d;
        d = '0;
        d[31:0]   = seed * 1000 + i;
        d[63:32]  = ~(seed * 1000 + i);
        d[95:64]  = (seed + 1) * 7 + i * 13;
        d[127:96] = 32'hA5A50000 + i;
        return d;
    endfunction

    // Send one beat on the weight stream, honouring tready.
    task automatic w_send(input [WORD_W-1:0] data, input logic last);
        @(negedge clk) begin
            w_tvalid = 1'b1;
            w_tdata  = data;
            w_tlast  = last;
        end
        // hold until the beat is actually accepted
        do @(posedge clk); while (w_tready !== 1'b1);
        @(negedge clk) begin
            w_tvalid = 1'b0;
            w_tlast  = 1'b0;
        end
    endtask

    task automatic a_send(input [WORD_W-1:0] data, input logic last);
        @(negedge clk) begin
            a_tvalid = 1'b1;
            a_tdata  = data;
            a_tlast  = last;
        end
        do @(posedge clk); while (a_tready !== 1'b1);
        @(negedge clk) begin
            a_tvalid = 1'b0;
            a_tlast  = 1'b0;
        end
    endtask

    task automatic w_readback(input int addr, output [WORD_W-1:0] data);
        @(negedge clk) begin w_rd_en = 1'b1; w_rd_addr = addr[AW-1:0]; end
        @(negedge clk) w_rd_en = 1'b0;
        @(posedge clk);
        data = w_rd_data;
    endtask

    task automatic a_readback(input int addr, output [WORD_W-1:0] data);
        @(negedge clk) begin a_rd_en = 1'b1; a_rd_addr = addr[AW-1:0]; end
        @(negedge clk) a_rd_en = 1'b0;
        @(posedge clk);
        data = a_rd_data;
    endtask

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    logic [WORD_W-1:0] rd;
    logic [WORD_W-1:0] exp;

    initial begin
        rst_n = 0;
        w_tvalid = 0; w_tdata = '0; w_tlast = 0; w_buf_ready = 1;
        a_tvalid = 0; a_tdata = '0; a_tlast = 0; a_buf_ready = 1;
        w_rd_en = 0; w_rd_addr = '0;
        a_rd_en = 0; a_rd_addr = '0;

        repeat (3) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        //--- T1: 16 beats with tlast, landing at 0..15 ----------------
        for (int i = 0; i < 16; i++)
            w_send(payload(1, i), (i == 15));

        repeat (2) @(negedge clk);
        begin
            int bad;
            bad = 0;
            if (w_writes !== 16) begin
                errors++; bad++;
                $error("[T1] adapter issued %0d writes, expected 16", w_writes);
            end
            for (int i = 0; i < 16; i++) begin
                w_readback(i, rd);
                exp = payload(1, i);
                if (rd !== exp) begin
                    bad++; errors++;
                    $error("[T1] buffer word %0d = %h, expected %h", i, rd, exp);
                end
                if (w_seen_addr[i] !== i) begin
                    bad++; errors++;
                    $error("[T1] write %0d went to addr %0d, expected %0d",
                           i, w_seen_addr[i], i);
                end
            end
            if (bad == 0)
                $display("PASS [T1] 16 beats landed sequentially at 0..15, read back exact");
        end

        //--- T2: tlast reset the counter -----------------------------
        // A second transfer must start at 0 again, not continue at 16.
        for (int i = 0; i < 4; i++)
            w_send(payload(2, i), (i == 3));

        repeat (2) @(negedge clk);
        begin
            int bad;
            bad = 0;
            for (int i = 0; i < 4; i++) begin
                w_readback(i, rd);
                exp = payload(2, i);
                if (rd !== exp) begin
                    bad++; errors++;
                    $error("[T2] word %0d = %h, expected the SECOND transfer's %h",
                           i, rd, exp);
                end
            end
            // words 4..15 must still hold transfer 1 - the second
            // transfer was shorter and must not have touched them
            for (int i = 4; i < 16; i++) begin
                w_readback(i, rd);
                exp = payload(1, i);
                if (rd !== exp) begin
                    bad++; errors++;
                    $error("[T2] word %0d = %h, expected transfer 1's %h (untouched)",
                           i, rd, exp);
                end
            end
            if (bad == 0)
                $display("PASS [T2] tlast reset the counter: transfer 2 restarted at 0, left 4..15 intact");
        end

        //--- T3: backpressure ----------------------------------------
        // Hold buf_ready low with tvalid high: tready must fall, no
        // write may occur, and the stream must resume exactly where it
        // left off. Checked by DATA, so a drop and a duplicate both
        // fail rather than cancelling out in a count.
        begin
            int bad;
            int writes_before;
            bad = 0;

            // beats 0..1 flow normally
            for (int i = 0; i < 2; i++)
                w_send(payload(3, i), 1'b0);

            // now stall: tvalid high, buf_ready low
            @(negedge clk) begin
                w_tvalid    = 1'b1;
                w_tdata     = payload(3, 2);
                w_tlast     = 1'b0;
                w_buf_ready = 1'b0;
            end
            @(posedge clk);
            if (w_tready !== 1'b0) begin
                bad++; errors++;
                $error("[T3] tready = %b while buf_ready low, expected 0", w_tready);
            end

            writes_before = w_writes;
            repeat (5) @(negedge clk);
            if (w_writes !== writes_before) begin
                bad++; errors++;
                $error("[T3] %0d writes occurred during the stall, expected 0",
                       w_writes - writes_before);
            end

            // release and let the held beat through
            @(negedge clk) w_buf_ready = 1'b1;
            do @(posedge clk); while (w_tready !== 1'b1);
            @(negedge clk) begin w_tvalid = 1'b0; w_tlast = 1'b0; end

            // remaining beats
            for (int i = 3; i < 8; i++)
                w_send(payload(3, i), (i == 7));

            repeat (2) @(negedge clk);
            for (int i = 0; i < 8; i++) begin
                w_readback(i, rd);
                exp = payload(3, i);
                if (rd !== exp) begin
                    bad++; errors++;
                    $error("[T3] word %0d = %h, expected %h (beat lost, duplicated or misplaced)",
                           i, rd, exp);
                end
            end
            if (bad == 0)
                $display("PASS [T3] backpressure: tready fell, zero writes while stalled, all 8 beats landed correctly");
        end

        //--- T4: wrap past DEPTH -------------------------------------
        // No tlast, more beats than the buffer holds: addresses wrap to
        // 0 and the sticky flag reports it instead of corrupting quietly.
        begin
            int bad;
            bad = 0;
            if (w_wrapped !== 1'b0) begin
                bad++; errors++;
                $error("[T4] wrapped already set before the oversized transfer");
            end

            for (int i = 0; i < DEPTH + 3; i++)
                w_send(payload(4, i), 1'b0);

            repeat (2) @(negedge clk);
            if (w_wrapped !== 1'b1) begin
                bad++; errors++;
                $error("[T4] wrapped = %b after %0d beats into a %0d-word buffer, expected 1",
                       w_wrapped, DEPTH + 3, DEPTH);
            end
            // the last 3 beats overwrote words 0..2
            for (int i = 0; i < 3; i++) begin
                w_readback(i, rd);
                exp = payload(4, DEPTH + i);
                if (rd !== exp) begin
                    bad++; errors++;
                    $error("[T4] word %0d = %h, expected the wrapped beat %h",
                           i, rd, exp);
                end
            end
            if (bad == 0)
                $display("PASS [T4] oversized transfer wrapped to 0 and set the sticky wrapped flag");
        end

        //--- T5: the activation instance, independently ---------------
        begin
            int bad;
            bad = 0;
            for (int i = 0; i < 12; i++)
                a_send(payload(5, i), (i == 11));

            repeat (2) @(negedge clk);
            if (a_writes !== 12) begin
                bad++; errors++;
                $error("[T5] activation adapter issued %0d writes, expected 12", a_writes);
            end
            for (int i = 0; i < 12; i++) begin
                a_readback(i, rd);
                exp = payload(5, i);
                if (rd !== exp) begin
                    bad++; errors++;
                    $error("[T5] activation word %0d = %h, expected %h", i, rd, exp);
                end
            end
            // the weight buffer must be untouched by activation traffic
            for (int i = 0; i < 3; i++) begin
                w_readback(i, rd);
                exp = payload(4, DEPTH + i);
                if (rd !== exp) begin
                    bad++; errors++;
                    $error("[T5] weight word %0d = %h, disturbed by activation traffic (expected %h)",
                           i, rd, exp);
                end
            end
            if (bad == 0)
                $display("PASS [T5] activation instance works independently; weight buffer undisturbed");
        end

        //--- summary --------------------------------------------------
        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
