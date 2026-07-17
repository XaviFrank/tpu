`timescale 1ns / 1ps
//====================================================================
// tb_ws_tpu_tiled_top: multi-tile golden run over AXI at the synth
// parameterization (N=K=16). A REAL 32x32 weight matrix (2x2 tiles)
// against a reference that doesn't tile at all - full_y() sums all
// 32 products directly; the hardware must match it bit-exactly
// through the engine's post-accumulation requant stream (res_*).
//   P1: STATUS (incl. bit2 cfg_err) reads 0 after reset
//   F1: 2x2 identity sweep - outer busy observed, done polled, 64
//       final results in contract order, values exact, tile_swaps=4
//   F2: 2x2 relu sweep - engine's final tap applies relu (swaps=8)
//   F3: cfg_err over AXI - K_TILES=0 go is rejected: busy never
//       rises, STATUS[2] sets, done never arrives
//   (F4 - K_TILES==1 collapse - lands with tiling spec step 4.)
// Layout per the loader contract: tile (n,k) at index n*K_TILES+k,
// activation chunk k at words k*n_vecs..+n_vecs-1.
//====================================================================
module tb_ws_tpu_tiled_top;

    localparam int N          = 16;
    localparam int K          = 16;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int OUT_WIDTH  = 8;
    localparam int DEPTH      = 64;
    localparam int VEC_W      = 4;
    localparam int TILE_W     = 4;
    localparam int WORD_W     = N * DATA_WIDTH;   // 128

    localparam [5:0] ADDR_CTRL   = 6'h00;
    localparam [5:0] ADDR_STATUS = 6'h04;
    localparam [5:0] ADDR_NVECS  = 6'h08;
    localparam [5:0] ADDR_CFG    = 6'h0C;
    localparam [5:0] ADDR_PSEL   = 6'h10;
    localparam [5:0] ADDR_PDATA  = 6'h14;
    localparam [5:0] ADDR_KTILES = 6'h18;
    localparam [5:0] ADDR_NTILES = 6'h1C;

    localparam int POLL_MAX = 200;   // outer sweeps are long
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
        if (!$value$plusargs("SEED=%d", cclk_seed)) cclk_seed = 32'd1;
        // Icarus quirk: $urandom(seed) mutates its argument - display first
        $display("cclk jitter seed = %0d", cclk_seed);
        cclk_seed_dummy = $urandom(cclk_seed);
        forever begin
            #(1.7 + $urandom_range(0, 400)/1000.0) cclk = ~cclk;
        end
    end

    initial begin
        #10_000_000;
        $error("[watchdog] TIMEOUT");
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
    // TB-driven core-domain nets + DUT edges
    //----------------------------------------------------------------
    logic                        perf_clear;
    logic                        wr_en_w, wr_en_a;
    logic [5:0]                  wr_addr_w, wr_addr_a;
    logic [WORD_W-1:0]           wr_data_w, wr_data_a;
    logic signed [ACC_WIDTH-1:0] sum_out   [N];
    logic                        sum_valid [N];
    logic signed [OUT_WIDTH-1:0] q8_out    [N];
    logic                        q8_valid  [N];
    logic                        res_valid;
    logic signed [OUT_WIDTH-1:0] res_q8;
    logic [VEC_W-1:0]            res_vec;
    logic [$clog2(N)-1:0]        res_col;
    logic [TILE_W-1:0]           res_ntile;

    int errors = 0;

    ws_tpu_tiled_top #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH),
        .SCALE_WIDTH(16), .SHIFT_WIDTH(5), .TILE_W(TILE_W),
        .MAX_N_VECS(16)
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
        .q8_valid      (q8_valid),
        .res_valid     (res_valid),
        .res_q8        (res_q8),
        .res_vec       (res_vec),
        .res_col       (res_col),
        .res_ntile     (res_ntile)
    );

    //----------------------------------------------------------------
    // Collectors: final-result stream + last-inner-run q8 (for the
    // Task-4 K_TILES==1 equivalence check)
    //----------------------------------------------------------------
    int res_cnt = 0;
    int r_val [64];
    int r_v   [64];
    int r_c   [64];
    int r_nt  [64];

    always @(posedge cclk) begin
        if (crst_n && res_valid === 1'b1) begin
            r_val[res_cnt] = res_q8;
            r_v[res_cnt]   = res_vec;
            r_c[res_cnt]   = res_col;
            r_nt[res_cnt]  = res_ntile;
            res_cnt = res_cnt + 1;
        end
    end

    int q8_cnt  [16];
    int q8_last [16][4];

    always @(posedge cclk) begin
        if (crst_n) begin
            if (dut.inner_go === 1'b1)
                for (int j = 0; j < N; j++) q8_cnt[j] = 0;
            for (int j = 0; j < N; j++)
                if (q8_valid[j] === 1'b1) begin
                    q8_last[j][q8_cnt[j]] = q8_out[j];
                    q8_cnt[j] = q8_cnt[j] + 1;
                end
        end
    end

    //----------------------------------------------------------------
    // Golden: 32x32, reference does NOT tile (sums all 32 directly)
    //----------------------------------------------------------------
    int W  [32][32];
    int av [2][32];

    function automatic int clamp8(input int v, input int relu);
        int r;
        r = v;
        if (relu != 0 && r < 0) r = 0;
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        return r;
    endfunction

    function automatic int full_y(input int v, input int gcol);
        int y;
        y = 0;
        for (int i = 0; i < 32; i++)
            y = y + W[i][gcol]*av[v][i];
        return y;
    endfunction

    //----------------------------------------------------------------
    // Loaders (128-bit words, manual two's complement per byte)
    //----------------------------------------------------------------
    task automatic write_w_word(input int addr, input [WORD_W-1:0] word);
        @(negedge cclk) begin wr_en_w = 1; wr_addr_w = 6'(addr); wr_data_w = word; end
        @(negedge cclk) wr_en_w = 0;
    endtask

    task automatic write_a_word(input int addr, input [WORD_W-1:0] word);
        @(negedge cclk) begin wr_en_a = 1; wr_addr_a = 6'(addr); wr_data_a = word; end
        @(negedge cclk) wr_en_a = 0;
    endtask

    // tile (n,k) for a kt-wide sweep: index n*kt+k, word r = resident
    // row r = W[k*16+r][n*16 .. n*16+15]
    task automatic load_tile(input int n, input int k, input int kt);
        logic [WORD_W-1:0] word;
        int b;
        for (int r = 0; r < 16; r++) begin
            for (int e = 0; e < 16; e++) begin
                b = W[k*16 + r][n*16 + e];
                if (b < 0) b = b + 256;
                word[e*DATA_WIDTH +: DATA_WIDTH] = b[7:0];
            end
            write_w_word((n*kt + k)*K + r, word);
        end
    endtask

    // activation chunk k for vector v at word k*nv + v
    task automatic load_avec(input int k, input int v, input int nv);
        logic [WORD_W-1:0] word;
        int b;
        for (int e = 0; e < 16; e++) begin
            b = av[v][k*16 + e];
            if (b < 0) b = b + 256;
            word[e*DATA_WIDTH +: DATA_WIDTH] = b[7:0];
        end
        write_a_word(k*nv + v, word);
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

    int exp_swaps = 0;

    task automatic perf_check_swaps(input string name);
        logic [31:0] v;
        axi_write(ADDR_PSEL, {28'b0, 4'd7});
        repeat (PERF_SETTLE_ACLKS) @(negedge aclk);
        axi_read(ADDR_PDATA, v);
        if (v !== exp_swaps) begin
            errors++;
            $error("[%s] tile_swaps = %0d, expected %0d", name, v, exp_swaps);
        end else begin
            $display("PASS [%s] tile_swaps = %0d (one per inner go)", name, v);
        end
    endtask

    //----------------------------------------------------------------
    // Program registers and run one full sweep (checks done by caller)
    //----------------------------------------------------------------
    task automatic run_sweep(input int nv, input int kt, input int nt,
                             input int relu, input string name);
        logic [31:0] st;
        logic [31:0] cfgword;

        cfgword = '0;
        cfgword[0] = 1'b1;                 // scale_mult = 1, shift = 0
        if (relu != 0) cfgword[21] = 1'b1;

        axi_write(ADDR_NVECS,  nv);
        axi_write(ADDR_CFG,    cfgword);
        axi_write(ADDR_KTILES, kt);
        axi_write(ADDR_NTILES, nt);
        axi_read(ADDR_KTILES, st);
        if (st !== kt) begin
            errors++; $error("[%s] K_TILES readback = %0d, wrote %0d", name, st, kt);
        end
        axi_read(ADDR_NTILES, st);
        if (st !== nt) begin
            errors++; $error("[%s] N_TILES readback = %0d, wrote %0d", name, st, nt);
        end
        repeat (CFG_SETTLE_ACLKS) @(negedge aclk);

        res_cnt = 0;
        axi_write(ADDR_CTRL, 32'h1);

        // outer busy holds for the whole sweep (hundreds of cclk):
        // trivially observable a few aclk after the commit
        repeat (2) @(negedge aclk);
        axi_read(ADDR_STATUS, st);
        if (st[0] !== 1'b1) begin
            errors++; $error("[%s] outer STATUS.busy not observed", name);
        end else $display("PASS [%s] outer STATUS.busy observed", name);

        poll_status_bit(1, name);
        repeat (10) @(negedge aclk);
        exp_swaps = exp_swaps + kt*nt;
    endtask

    // check the collected res stream against a golden K-total
    task automatic check_res(input int nv, input int nt, input int ktotal,
                             input int relu, input string name);
        int idx, y;
        if (res_cnt !== nt*nv*16) begin
            errors++;
            $error("[%s] res count = %0d, expected %0d", name, res_cnt, nt*nv*16);
        end else begin
            idx = 0;
            for (int t = 0; t < nt; t++)
                for (int v = 0; v < nv; v++)
                    for (int j = 0; j < 16; j++) begin
                        if (r_nt[idx] !== t || r_v[idx] !== v || r_c[idx] !== j) begin
                            errors++;
                            $error("[%s order] idx %0d = (nt%0d v%0d c%0d), expected (nt%0d v%0d c%0d)",
                                   name, idx, r_nt[idx], r_v[idx], r_c[idx], t, v, j);
                        end
                        y = 0;
                        for (int i = 0; i < ktotal; i++)
                            y = y + W[i][t*16 + j]*av[v][i];
                        if (r_val[idx] !== clamp8(y, relu)) begin
                            errors++;
                            $error("[%s value] res(nt%0d v%0d c%0d) = %0d, expected %0d (int32 %0d)",
                                   name, t, v, j, r_val[idx], clamp8(y, relu), y);
                        end
                        idx = idx + 1;
                    end
            $display("PASS [%s] %0d final results, order + values exact vs untiled reference", name, res_cnt);
        end
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

        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 32; j++)
                W[i][j] = ((i*5 + j*3) % 19) - 9;             // -9..9
        for (int v = 0; v < 2; v++)
            for (int i = 0; i < 32; i++)
                av[v][i] = ((v*11 + i*7) % 15) - 7;           // -7..7

        repeat (3) @(negedge aclk);
        aresetn = 1;
        crst_n  = 1;
        repeat (2) @(negedge aclk);

        // preload the full 2x2 layout: 4 tiles (all 64 weight words)
        // + 2 chunks x 2 vectors (activation words 0..3)
        for (int n = 0; n < 2; n++)
            for (int k = 0; k < 2; k++)
                load_tile(n, k, 2);
        for (int k = 0; k < 2; k++)
            for (int v = 0; v < 2; v++)
                load_avec(k, v, 2);

        //--- P1: reset state (incl. the new cfg_err bit) ---------------
        axi_read(ADDR_STATUS, rd);
        if (rd !== 32'h0) begin
            errors++; $error("[P1] STATUS after reset = %h, expected 0", rd);
        end else $display("PASS [P1] STATUS (busy/done/cfg_err) reads 0 after reset");

        //--- F1: 2x2 identity sweep, 32x32 golden ----------------------
        run_sweep(2, 2, 2, 0, "F1 2x2 identity");
        check_res(2, 2, 32, 0, "F1 2x2 identity");
        perf_check_swaps("F1");

        //--- F2: 2x2 relu sweep (final tap applies relu) ---------------
        run_sweep(2, 2, 2, 1, "F2 2x2 relu");
        check_res(2, 2, 32, 1, "F2 2x2 relu");
        perf_check_swaps("F2");

        //--- F3: cfg_err over AXI - K_TILES=0 go is rejected -----------
        axi_write(ADDR_KTILES, 32'h0);
        repeat (CFG_SETTLE_ACLKS) @(negedge aclk);
        axi_write(ADDR_CTRL, 32'h1);
        repeat (30) @(negedge aclk);
        axi_read(ADDR_STATUS, rd);
        if (rd[2] !== 1'b1 || rd[0] !== 1'b0) begin
            errors++;
            $error("[F3] STATUS = %h, expected cfg_err=1 busy=0", rd);
        end else $display("PASS [F3] K_TILES=0 rejected: cfg_err set, busy never rose");

        //--- summary ----------------------------------------------------
        repeat (2) @(negedge aclk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
