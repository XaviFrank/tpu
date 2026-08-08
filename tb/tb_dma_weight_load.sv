`timescale 1ns / 1ps
//====================================================================
// tb_dma_weight_load: the DMA delivery path, in simulation, with the
// CORRECTED buf_ready wiring.
//
//   axi_dma's M_AXIS_MM2S -> axi_stream_to_buf_adapter -> ws_tpu_top's
//   raw weight write port, with buf_ready = !core_busy_out instead of
//   the block design's xlconstant(1).
//
//   WHY THIS TB HAD TO EXIST. The adapter's own TB drives buf_ready as
//   a stimulus signal, so it proves the adapter RESPECTS backpressure -
//   it cannot prove anything about where backpressure comes from,
//   because it has no ws_tpu_top in it. And no RTL module instantiates
//   both, so before this file the "DMA path" existed only as a Vivado
//   block design: unsimulatable, and the one place the constant-1
//   tie-off lived. A port added to ws_tpu_top and verified by nothing
//   would be the worst version of this change.
//
//   The DMA IP itself is NOT modelled. What it emits is AXI4-Stream -
//   tvalid/tready/tdata/tlast at 128 bits, per C_M_AXIS_MM2S_TDATA_WIDTH
//   in the block design - and that is driven directly here. Modelling
//   Xilinx's DMA would test Xilinx's DMA.
//
//   D1  16 weight words delivered as stream beats only, peeked back
//       through AXI-Lite. Peek shares the controller's real buffer
//       read port, so this is the tile_buffer, not a shadow copy.
//   D2  a run on stream-delivered weights reproduces the golden -
//       the bits are not just present, they are consumed correctly.
//   D3  THE POINT OF THE CHANGE: with a run in flight, tvalid is held
//       high and tready must stay low, no beat may be accepted, and
//       the weight tile must be byte-identical afterwards. Under the
//       old constant-1 tie this test fails - the beats land.
//   D4  the stream resumes and re-delivers once busy drops, so the
//       backpressure is a stall and not a deadlock.
//
//   Activations still arrive by poke. That is the real division on
//   this build - the adapter instance in the block design is wired to
//   wr_*_w only - and inventing a second adapter here would test a
//   configuration that is not on the board.
//====================================================================
module tb_dma_weight_load;

    localparam int N          = 16;
    localparam int K          = 16;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int OUT_WIDTH  = 8;
    localparam int DEPTH      = 64;
    localparam int VEC_W      = 4;
    localparam int WORD_W     = N * DATA_WIDTH;   // 128, = the DMA's tdata
    localparam int LANES      = WORD_W / 32;      // 4
    localparam int NROWS      = 16;               // weight tile words
    localparam int NVECS      = 2;

    localparam [5:0] ADDR_CTRL   = 6'h00;
    localparam [5:0] ADDR_STATUS = 6'h04;
    localparam [5:0] ADDR_NVECS  = 6'h08;
    localparam [5:0] ADDR_CFG    = 6'h0C;
    localparam [5:0] ADDR_POKEA  = 6'h20;
    localparam [5:0] ADDR_POKED  = 6'h24;
    localparam [5:0] ADDR_PEEKA  = 6'h28;
    localparam [5:0] ADDR_PEEKD  = 6'h2C;

    localparam int PEEK_SETTLE_ACLKS = 20;
    localparam int POLL_MAX          = 400;

    //----------------------------------------------------------------
    // Two clocks. The adapter and the core share ONE (pl_clk1 in the
    // block design), which is the whole reason core_busy_out can be
    // exported unsynchronized - so they share one here too.
    //----------------------------------------------------------------
    logic aclk = 1'b0;
    logic cclk = 1'b0;
    logic aresetn, crst_n;

    always #5   aclk = ~aclk;    // 100 MHz AXI-Lite
    always #3.2 cclk = ~cclk;    // core / stream clock

    int errors = 0;

    initial begin
        #20_000_000;
        $display("Error: [watchdog] TIMEOUT");
        $display("FAILED");
        $finish;
    end

    //----------------------------------------------------------------
    // AXI-Lite
    //----------------------------------------------------------------
    logic        awvalid, awready, wvalid, wready, bvalid, bready;
    logic [5:0]  awaddr;
    logic [31:0] wdata;
    logic [3:0]  wstrb;
    logic [1:0]  bresp;
    logic        arvalid, arready, rvalid, rready;
    logic [5:0]  araddr;
    logic [31:0] rdata;
    logic [1:0]  rresp;

    //----------------------------------------------------------------
    // Stream + adapter + the wiring under test
    //----------------------------------------------------------------
    logic                  s_tvalid, s_tready, s_tlast;
    logic [WORD_W-1:0]     s_tdata;
    logic                  buf_ready, wrapped;
    logic                  wr_en_w;
    logic [5:0]            wr_addr_w;
    logic [WORD_W-1:0]     wr_data_w;

    logic                  wr_en_a;
    logic [5:0]            wr_addr_a;
    logic [WORD_W-1:0]     wr_data_a;
    logic                  perf_clear;
    logic                  core_busy_out;
    logic signed [ACC_WIDTH-1:0] sum_out   [N];
    logic                        sum_valid [N];
    logic signed [OUT_WIDTH-1:0] q8_out    [N];
    logic                        q8_valid  [N];

    // THE CHANGE UNDER TEST. Same domain, so no synchronizer.
    assign buf_ready = !core_busy_out;

    axi_stream_to_buf_adapter #(.WORD_W(WORD_W), .DEPTH(DEPTH)) u_adapt (
        .clk       (cclk),
        .rst_n     (crst_n),
        .s_tvalid  (s_tvalid),
        .s_tready  (s_tready),
        .s_tdata   (s_tdata),
        .s_tlast   (s_tlast),
        .buf_ready (buf_ready),
        .wr_en     (wr_en_w),
        .wr_addr   (wr_addr_w),
        .wr_data   (wr_data_w),
        .wrapped   (wrapped)
    );

    ws_tpu_top #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH),
        .SCALE_WIDTH(16), .SHIFT_WIDTH(5), .DEBUG_CAPTURE(1)
    ) dut (
        .s_axi_aclk    (aclk),
        .s_axi_aresetn (aresetn),
        .s_axi_awvalid (awvalid),
        .s_axi_awready (awready),
        .s_axi_awaddr  (awaddr),
        .s_axi_wvalid  (wvalid),
        .s_axi_wready  (wready),
        .s_axi_wdata   (wdata),
        .s_axi_wstrb   (wstrb),
        .s_axi_bvalid  (bvalid),
        .s_axi_bready  (bready),
        .s_axi_bresp   (bresp),
        .s_axi_arvalid (arvalid),
        .s_axi_arready (arready),
        .s_axi_araddr  (araddr),
        .s_axi_rvalid  (rvalid),
        .s_axi_rready  (rready),
        .s_axi_rdata   (rdata),
        .s_axi_rresp   (rresp),
        .core_clk      (cclk),
        .core_rst_n    (crst_n),
        .core_busy_out (core_busy_out),
        .wr_en_w       (wr_en_w),
        .wr_addr_w     (wr_addr_w),
        .wr_data_w     (wr_data_w),
        .wr_en_a       (wr_en_a),
        .wr_addr_a     (wr_addr_a),
        .wr_data_a     (wr_data_a),
        .perf_clear    (perf_clear),
        .sum_out       (sum_out),
        .sum_valid     (sum_valid),
        .q8_out        (q8_out),
        .q8_valid      (q8_valid)
    );

    //----------------------------------------------------------------
    // Data. Distinct bytes spanning both signs, so a lane-order or a
    // sign-extension bug cannot alias into a pass.
    //----------------------------------------------------------------
    int W  [16][16];
    int av [NVECS][16];

    function automatic [WORD_W-1:0] pack_row(input int r);
        logic [WORD_W-1:0] word;
        int b;
        word = '0;
        for (int e = 0; e < 16; e++) begin
            b = W[r][e];
            if (b < 0) b = b + 256;
            word[e*DATA_WIDTH +: DATA_WIDTH] = b[7:0];
        end
        return word;
    endfunction

    function automatic [WORD_W-1:0] pack_avec(input int v);
        logic [WORD_W-1:0] word;
        int b;
        word = '0;
        for (int e = 0; e < 16; e++) begin
            b = av[v][e];
            if (b < 0) b = b + 256;
            word[e*DATA_WIDTH +: DATA_WIDTH] = b[7:0];
        end
        return word;
    endfunction

    function automatic int golden_sum(input int v, input int j);
        int y;
        y = 0;
        for (int i = 0; i < 16; i++) y = y + W[i][j]*av[v][i];
        return y;
    endfunction

    //----------------------------------------------------------------
    // AXI-Lite BFM (drive negedge, sample posedge)
    //----------------------------------------------------------------
    task automatic axi_write(input [5:0] addr, input [31:0] data);
        @(negedge aclk) begin
            awvalid = 1; awaddr = addr;
            wvalid  = 1; wdata  = data;
            bready  = 1;
        end
        fork
            begin
                do @(posedge aclk); while (awready !== 1'b1);
                @(negedge aclk) awvalid = 0;
            end
            begin
                do @(posedge aclk); while (wready !== 1'b1);
                @(negedge aclk) wvalid = 0;
            end
        join
        do @(posedge aclk); while (bvalid !== 1'b1);
        if (bresp !== 2'b00) begin
            errors++; $display("Error: axi_write bresp = %b", bresp);
        end
        @(negedge aclk) bready = 0;
    endtask

    task automatic axi_read(input [5:0] addr, output [31:0] data);
        @(negedge aclk) begin arvalid = 1; araddr = addr; rready = 1; end
        do @(posedge aclk); while (arready !== 1'b1);
        @(negedge aclk) arvalid = 0;
        do @(posedge aclk); while (rvalid !== 1'b1);
        data = rdata;
        if (rresp !== 2'b00) begin
            errors++; $display("Error: axi_read rresp = %b", rresp);
        end
        @(negedge aclk) rready = 0;
    endtask

    task automatic wait_poke_idle;
        logic [31:0] v;
        int n;
        n = 0;
        forever begin
            axi_read(ADDR_STATUS, v);
            if (v[3] === 1'b0) return;
            n = n + 1;
            if (n == POLL_MAX) begin
                errors++; $display("Error: poke_busy never cleared"); return;
            end
        end
    endtask

    task automatic poke_word(input int sel, input int addr,
                             input [WORD_W-1:0] word);
        logic [31:0] a;
        for (int lane = 0; lane < LANES; lane++) begin
            a = '0;
            a[5:0] = 6'(addr);
            a[7:6] = 2'(lane);
            a[8]   = 1'(sel);
            axi_write(ADDR_POKEA, a);
            if (lane == LANES-1) wait_poke_idle();
            axi_write(ADDR_POKED, word[lane*32 +: 32]);
        end
    endtask

    task automatic peek_elem(input int buf_sel, input int idx1,
                             input int idx0, output [31:0] val);
        logic [31:0] a;
        a = '0;
        a[5:0]   = 6'(idx0);
        a[9:6]   = 4'(idx1);
        a[11:10] = 2'(buf_sel);
        axi_write(ADDR_PEEKA, a);
        repeat (PEEK_SETTLE_ACLKS) @(negedge aclk);
        axi_read(ADDR_PEEKD, val);
    endtask

    // Read one weight word back out of the real tile_buffer.
    task automatic peek_wword(input int addr, output [WORD_W-1:0] word);
        logic [31:0] v;
        word = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            peek_elem(0, lane, addr, v);
            word[lane*32 +: 32] = v;
        end
    endtask

    task automatic poll_done;
        logic [31:0] v;
        int n;
        n = 0;
        forever begin
            axi_read(ADDR_STATUS, v);
            if (v[1] === 1'b1) return;
            n = n + 1;
            if (n == POLL_MAX) begin
                errors++; $display("Error: done never asserted"); return;
            end
        end
    endtask

    //----------------------------------------------------------------
    // AXI4-Stream source. One beat per accepted cycle; tvalid is held
    // until tready, which is what a real DMA does under backpressure.
    //----------------------------------------------------------------
    task automatic stream_tile;
        begin
            for (int r = 0; r < NROWS; r++) begin
                @(negedge cclk);
                s_tvalid = 1'b1;
                s_tdata  = pack_row(r);
                s_tlast  = (r == NROWS-1);
                do @(posedge cclk); while (s_tready !== 1'b1);
            end
            @(negedge cclk);
            s_tvalid = 1'b0;
            s_tlast  = 1'b0;
        end
    endtask

    logic [WORD_W-1:0] got, exp;
    logic [31:0]       v32;
    int bad, gs, accepted_during_run;

    // Independent witness: count every beat the adapter accepts. D3
    // needs "no beat was taken", which a data comparison alone cannot
    // show - re-delivering identical data would look the same.
    always_ff @(posedge cclk) begin
        if (crst_n && s_tvalid && s_tready) begin
            if (core_busy_out) accepted_during_run <= accepted_during_run + 1;
        end
    end

    initial begin
        awvalid=0; wvalid=0; bready=0; arvalid=0; rready=0;
        awaddr=0; wdata=0; wstrb=4'hF; araddr=0;
        s_tvalid=0; s_tdata='0; s_tlast=0;
        wr_en_a=0; wr_addr_a=0; wr_data_a='0; perf_clear=0;
        accepted_during_run = 0;
        aresetn = 0; crst_n = 0;

        for (int r = 0; r < 16; r++)
            for (int c = 0; c < 16; c++)
                W[r][c] = ((r*16 + c*7 + 3) % 255) - 127;
        for (int v = 0; v < NVECS; v++)
            for (int e = 0; e < 16; e++)
                av[v][e] = ((v*53 + e*11 + 5) % 255) - 127;

        repeat (8) @(negedge aclk);
        aresetn = 1; crst_n = 1;
        repeat (8) @(negedge aclk);

        $display("=== dma_weight_load: stream -> adapter -> ws_tpu_top ===");

        // ---- D1: deliver the tile by stream only ------------------
        stream_tile();
        repeat (4) @(negedge cclk);

        bad = 0;
        for (int r = 0; r < NROWS; r++) begin
            peek_wword(r, got);
            exp = pack_row(r);
            if (got !== exp) begin
                bad = bad + 1;
                if (bad <= 3)
                    $display("Error: D1 word %0d got %032h exp %032h", r, got, exp);
            end
        end
        errors = errors + bad;
        if (wrapped !== 1'b0) begin
            errors++;
            $display("Error: D1 wrapped asserted on a %0d-beat transfer", NROWS);
        end
        $display("  D1 %0d/%0d stream-delivered weight words exact, wrapped %b",
                 NROWS - bad, NROWS, wrapped);

        // ---- D2: a real run consumes them -------------------------
        for (int v = 0; v < NVECS; v++) poke_word(1, v, pack_avec(v));
        wait_poke_idle();
        axi_write(ADDR_NVECS, NVECS);
        axi_write(ADDR_CFG,   32'h0000_0001);   // mult=1, shift=0, relu off
        repeat (4) @(negedge aclk);
        axi_write(ADDR_CTRL,  32'h1);
        poll_done();

        bad = 0;
        for (int v = 0; v < NVECS; v++)
            for (int j = 0; j < 16; j++) begin
                peek_elem(3, j, v, v32);
                gs = golden_sum(v, j);
                if ($signed(v32) !== gs) begin
                    bad = bad + 1;
                    if (bad <= 3)
                        $display("Error: D2 y[%0d][%0d] got %0d exp %0d",
                                 v, j, $signed(v32), gs);
                end
            end
        errors = errors + bad;
        $display("  D2 %0d/%0d results match the golden on stream-delivered weights",
                 NVECS*16 - bad, NVECS*16);

        // ---- D3: backpressure while a run is in flight ------------
        // Restart a run, then push a DIFFERENT tile at the adapter
        // while busy. Nothing may be accepted and the tile must be
        // untouched. Under buf_ready = 1 these beats land.
        for (int r = 0; r < 16; r++)
            for (int c = 0; c < 16; c++)
                W[r][c] = 127 - W[r][c];        // a visibly different tile

        axi_write(ADDR_CTRL, 32'h1);            // go
        // hold tvalid high across the whole run
        fork
            begin : drive
                for (int r = 0; r < NROWS; r++) begin
                    @(negedge cclk);
                    s_tvalid = 1'b1;
                    s_tdata  = pack_row(r);
                    s_tlast  = (r == NROWS-1);
                    do @(posedge cclk); while (s_tready !== 1'b1);
                end
                @(negedge cclk);
                s_tvalid = 1'b0;
                s_tlast  = 1'b0;
            end
            begin : watch
                int seen_busy;
                seen_busy = 0;
                repeat (400) begin
                    @(posedge cclk);
                    if (core_busy_out) begin
                        seen_busy = seen_busy + 1;
                        if (s_tready !== 1'b0) begin
                            errors++;
                            $display("Error: D3 tready high while busy");
                            disable watch;
                        end
                    end
                end
                if (seen_busy == 0) begin
                    errors++;
                    $display("Error: D3 never observed busy - the test is vacuous");
                end else begin
                    $display("  D3 busy for %0d core cycles, tready low throughout",
                             seen_busy);
                end
            end
        join
        poll_done();
        repeat (4) @(negedge cclk);

        if (accepted_during_run != 0) begin
            errors++;
            $display("Error: D3 %0d beats accepted while busy", accepted_during_run);
        end
        $display("  D3 beats accepted while busy: %0d", accepted_during_run);

        // ---- D4: the stall was a stall, not a deadlock ------------
        // The fork above ran the whole tile to completion once busy
        // dropped, so the NEW tile must now be resident.
        bad = 0;
        for (int r = 0; r < NROWS; r++) begin
            peek_wword(r, got);
            exp = pack_row(r);
            if (got !== exp) begin
                bad = bad + 1;
                if (bad <= 3)
                    $display("Error: D4 word %0d got %032h exp %032h", r, got, exp);
            end
        end
        errors = errors + bad;
        $display("  D4 stream resumed after busy: %0d/%0d words of the new tile exact",
                 NROWS - bad, NROWS);

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

endmodule
