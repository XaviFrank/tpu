`timescale 1ns / 1ps
//====================================================================
// tb_ws_tpu_n16: N=16 scaling checkpoint for the CHIP-LEVEL top
// (verification, not new architecture). Runs ws_tpu_top at the exact
// parameterization the Vivado synth uses (N=16 / K=16, 128-bit
// buffer words, generics-over-defaults) across genuinely
// asynchronous clocks, entirely through the AXI-Lite register file:
//   F1: 4-vector identity-quant run - STATUS.busy observed mid-run,
//       done polled, exactly one core_go pulse, raw + q8 results
//       exact vs a 16x16 golden, perf load/stream/flush/tile_swaps
//       exact via PERF_SEL/PERF_DATA (N=16 budgets: 18/nv+1/33/1).
//   F2: relu run - REQUANT_CFG reaches the 16 requant lanes
//       (negative int32 results clamp to 0 at the int8 edge).
// Protocol corners (staggered AW/W, held-RREADY, torn-value watch)
// are N-independent and stay in tb_axi_lite_ctrl (N=4); this TB
// certifies that the integration scales, mirroring what tb_ws_n16
// does for the bare core.
// Single-tile scope: ws_tpu_top ties w_tile_idx/a_base to 0, so the
// weight tile lives at words 0..15 and vectors at words 0..3.
//====================================================================
module tb_ws_tpu_n16;

    localparam int N          = 16;
    localparam int K          = 16;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int OUT_WIDTH  = 8;
    localparam int DEPTH      = 64;   // 4 tiles of 16 words
    localparam int VEC_W      = 4;
    localparam int WORD_W     = N * DATA_WIDTH;   // 128

    localparam [5:0] ADDR_CTRL   = 6'h00;
    localparam [5:0] ADDR_STATUS = 6'h04;
    localparam [5:0] ADDR_NVECS  = 6'h08;
    localparam [5:0] ADDR_CFG    = 6'h0C;
    localparam [5:0] ADDR_PSEL   = 6'h10;
    localparam [5:0] ADDR_PDATA  = 6'h14;

    localparam int POLL_MAX = 50;
    // crossing-settle waits (round trip ~4-6 aclk + ~6 cclk actual;
    // 20 aclk budget per the axi_lite_ctrl HOST CONTRACT)
    localparam int CFG_SETTLE_ACLKS  = 20;
    localparam int PERF_SETTLE_ACLKS = 20;

    //----------------------------------------------------------------
    // Clocks + resets (two genuinely asynchronous domains)
    //----------------------------------------------------------------
    logic aclk = 1'b0;
    logic cclk = 1'b0;
    logic aresetn, crst_n;
    int unsigned cclk_seed;
    int unsigned cclk_seed_dummy;

    always #5 aclk = ~aclk;    // 100 MHz AXI clock

    initial begin
        // default seed keeps regressions reproducible; pass +SEED=<n>
        // to vvp to sweep a different phase trajectory.
        if (!$value$plusargs("SEED=%d", cclk_seed)) cclk_seed = 32'd1;
        // Icarus quirk: $urandom(seed) treats seed as inout, so the
        // seed must be displayed BEFORE the reseed call.
        $display("cclk jitter seed = %0d", cclk_seed);
        cclk_seed_dummy = $urandom(cclk_seed);
        forever begin
            #(1.7 + $urandom_range(0, 400)/1000.0) cclk = ~cclk;
        end
    end

    // watchdog: a hung crossing must fail the regression, not wedge it
    initial begin
        #5_000_000;
        $error("[watchdog] TIMEOUT - run never finished");
        $finish;
    end

    //----------------------------------------------------------------
    // AXI-Lite signals
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
    // TB-driven core-domain nets (raw write ports = future DMA path)
    //----------------------------------------------------------------
    logic                        perf_clear;   // tied 0 (no AXI clear bit)
    logic                        wr_en_w, wr_en_a;
    logic [5:0]                  wr_addr_w, wr_addr_a;
    logic [WORD_W-1:0]           wr_data_w, wr_data_a;
    logic signed [ACC_WIDTH-1:0] sum_out   [N];
    logic                        sum_valid [N];
    logic signed [OUT_WIDTH-1:0] q8_out    [N];
    logic                        q8_valid  [N];

    int errors  = 0;
    int go_seen = 0;
    int go_base = 0;

    ws_tpu_top #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH),
        .SCALE_WIDTH(16), .SHIFT_WIDTH(5)
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
    // Core-domain monitors: go pulse counter + raw/q8 collectors
    //----------------------------------------------------------------
    always @(posedge cclk)
        if (crst_n && dut.core_go === 1'b1) go_seen = go_seen + 1;

    int got8    [16][4];
    int got_raw [16][4];
    int got_cnt [16];

    always @(posedge cclk) begin
        if (crst_n) begin
            for (int j = 0; j < N; j++) begin
                if (q8_valid[j] === 1'b1) begin
                    got8[j][got_cnt[j]]    = q8_out[j];
                    got_raw[j][got_cnt[j]] = sum_out[j];
                    got_cnt[j] = got_cnt[j] + 1;
                end
            end
        end
    end

    //----------------------------------------------------------------
    // Deterministic golden data (same formulas as tb_ws_n16)
    //----------------------------------------------------------------
    int w  [16][16];   // resident weights w[i][j]
    int av [4][16];    // av[v][i]

    function automatic int clamp8(input int v, input int relu);
        int r;
        r = v;
        if (relu != 0 && r < 0) r = 0;
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        return r;
    endfunction

    function automatic int matvec(input int v, input int j);
        int y;
        y = 0;
        for (int i = 0; i < N; i++)
            y = y + w[i][j]*av[v][i];
        return y;
    endfunction

    //----------------------------------------------------------------
    // Loaders: one word = 16 packed int8s, element 0 in [7:0];
    // manual two's complement per byte. Tile 0 / base 0 (the tie-off).
    //----------------------------------------------------------------
    task automatic write_w_row(input int r);
        logic [WORD_W-1:0] word;
        int b;
        for (int e = 0; e < N; e++) begin
            b = w[r][e];
            if (b < 0) b = b + 256;
            word[e*DATA_WIDTH +: DATA_WIDTH] = b[7:0];
        end
        @(negedge cclk) begin
            wr_en_w = 1; wr_addr_w = 6'(r); wr_data_w = word;
        end
        @(negedge cclk) wr_en_w = 0;
    endtask

    task automatic write_a_vec(input int v);
        logic [WORD_W-1:0] word;
        int b;
        for (int e = 0; e < N; e++) begin
            b = av[v][e];
            if (b < 0) b = b + 256;
            word[e*DATA_WIDTH +: DATA_WIDTH] = b[7:0];
        end
        @(negedge cclk) begin
            wr_en_a = 1; wr_addr_a = 6'(v); wr_data_a = word;
        end
        @(negedge cclk) wr_en_a = 0;
    endtask

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
            errors++; $error("axi_write: bresp = %b", bresp);
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
            errors++; $error("axi_read: rresp = %b", rresp);
        end
        @(negedge aclk) rready = 0;
    endtask

    task automatic poll_status_bit(input int bit_idx, input string name);
        logic [31:0] v;
        int n;
        n = 0;
        forever begin
            axi_read(ADDR_STATUS, v);
            if (v[bit_idx] === 1'b1) return;
            n = n + 1;
            if (n == POLL_MAX) begin
                errors++;
                $error("poll %s: STATUS bit %0d never set", name, bit_idx);
                return;
            end
        end
    endtask

    //----------------------------------------------------------------
    // Perf readback over AXI; cumulative expectations, N=16 budgets:
    // load = N+2 = 18, stream = nv+1, flush = 2N+1 = 33, 1 swap/go
    //----------------------------------------------------------------
    int exp_load = 0, exp_stream = 0, exp_flush = 0, exp_swaps = 0;

    task automatic perf_read(input [3:0] idx, output [31:0] val);
        axi_write(ADDR_PSEL, {28'b0, idx});
        repeat (PERF_SETTLE_ACLKS) @(negedge aclk);
        axi_read(ADDR_PDATA, val);
    endtask

    task automatic perf_check(input [3:0] idx, input int exp,
                              input string name);
        logic [31:0] v;
        perf_read(idx, v);
        if (v !== exp) begin
            errors++;
            $error("[%s] perf[%0d] = %0d, expected %0d", name, idx, v, exp);
        end else begin
            $display("PASS [%s] perf[%0d] = %0d", name, idx, v);
        end
    endtask

    //----------------------------------------------------------------
    // One full run through the register file at N=16
    //----------------------------------------------------------------
    int y_exp;

    task automatic run_via_axi(input int nv, input int relu,
                               input int check_busy, input string name);
        logic [31:0] st;
        logic [31:0] cfgword;

        for (int j = 0; j < N; j++) got_cnt[j] = 0;

        cfgword = '0;
        cfgword[0] = 1'b1;                 // scale_mult = 1, shift = 0
        if (relu != 0) cfgword[21] = 1'b1; // relu_en

        axi_write(ADDR_NVECS, nv);
        axi_write(ADDR_CFG, cfgword);
        axi_read(ADDR_NVECS, st);          // AXI-side readback sanity
        if (st !== nv) begin
            errors++; $error("[%s] N_VECS readback = %0d, wrote %0d", name, st, nv);
        end
        repeat (CFG_SETTLE_ACLKS) @(negedge aclk);   // let values cross

        go_base = go_seen;
        axi_write(ADDR_CTRL, 32'h1);

        if (check_busy != 0) begin
            // N=16 run holds busy 3N+4+nv = 52+nv cclk (~200+ ns =
            // ~21 aclk wide); busy_axi visible ~4 aclk after commit,
            // so sampling ~5-6 aclk in lands comfortably mid-window
            repeat (2) @(negedge aclk);
            axi_read(ADDR_STATUS, st);
            if (st[0] !== 1'b1) begin
                errors++; $error("[%s] STATUS.busy not observed mid-run", name);
            end else $display("PASS [%s] STATUS.busy observed", name);
        end

        poll_status_bit(1, name);
        repeat (10) @(negedge aclk);

        if (go_seen - go_base !== 1) begin
            errors++;
            $error("[%s] core_go pulses = %0d, expected 1", name, go_seen - go_base);
        end

        for (int j = 0; j < N; j++) begin
            if (got_cnt[j] !== nv) begin
                errors++;
                $error("[%s] column %0d: %0d results, expected %0d",
                       name, j, got_cnt[j], nv);
            end else begin
                for (int v = 0; v < nv; v++) begin
                    y_exp = matvec(v, j);
                    if (got_raw[j][v] !== y_exp) begin
                        errors++;
                        $error("[%s] raw y%0d[%0d] = %0d, expected %0d",
                               name, v, j, got_raw[j][v], y_exp);
                    end
                    if (got8[j][v] !== clamp8(y_exp, relu)) begin
                        errors++;
                        $error("[%s] q8 y%0d[%0d] = %0d, expected %0d (int32 %0d)",
                               name, v, j, got8[j][v], clamp8(y_exp, relu), y_exp);
                    end
                end
            end
        end
        $display("PASS [%s] %0d vectors x %0d columns checked (raw+q8)", name, nv, N);

        exp_load   = exp_load + (N + 2);
        exp_stream = exp_stream + (nv + 1);
        exp_flush  = exp_flush + (2*N + 1);
        exp_swaps  = exp_swaps + 1;
        perf_check(4'd1, exp_load,   name);
        perf_check(4'd2, exp_stream, name);
        perf_check(4'd3, exp_flush,  name);
        perf_check(4'd7, exp_swaps,  name);
    endtask

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    logic [31:0] rd;

    initial begin
        aresetn = 0; crst_n = 0;
        awvalid = 0; awaddr = '0; wvalid = 0; wdata = '0; wstrb = 4'hF;
        bready = 0; arvalid = 0; araddr = '0; rready = 0;
        perf_clear = 0;
        wr_en_w = 0; wr_addr_w = '0; wr_data_w = '0;
        wr_en_a = 0; wr_addr_a = '0; wr_data_a = '0;

        // deterministic fill, same formulas certified in tb_ws_n16:
        // 16-term dot products reach ~±600, clipping both ways
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                w[i][j] = ((i*5 + j*3) % 19) - 9;            // -9..9
        for (int v = 0; v < 4; v++)
            for (int i = 0; i < N; i++)
                av[v][i] = ((v*11 + i*7) % 15) - 7;          // -7..7

        repeat (3) @(negedge aclk);
        aresetn = 1;
        crst_n  = 1;
        repeat (2) @(negedge aclk);

        // preload: tile 0 (words 0..15), 4 vectors (words 0..3) -
        // ws_tpu_top ties w_tile_idx/a_base to 0
        for (int r = 0; r < N; r++)
            write_w_row(r);
        for (int v = 0; v < 4; v++)
            write_a_vec(v);

        //--- sanity: STATUS reads 0 after reset ------------------------
        axi_read(ADDR_STATUS, rd);
        if (rd !== 32'h0) begin
            errors++; $error("[P1] STATUS after reset = %h, expected 0", rd);
        end else $display("PASS [P1] STATUS reads 0 after reset");

        //--- F1: 4-vector identity run, busy observed, perf exact ------
        run_via_axi(4, 0, 1, "F1 N=16 identity 4-vec");

        //--- F2: relu run - config reaches all 16 lanes -----------------
        run_via_axi(1, 1, 0, "F2 N=16 relu 1-vec");

        //--- summary ----------------------------------------------------
        repeat (2) @(negedge aclk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
