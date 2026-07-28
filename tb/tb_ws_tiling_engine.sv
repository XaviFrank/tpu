`timescale 1ns / 1ps
//====================================================================
// tb_ws_tiling_engine: unit TB for the WS tiling engine (N=4, single
// clock - the engine is a core-domain module; CDC is axi_lite_ctrl's
// job and is not re-verified here). Engine + REAL ws_quant_unit.
//   T1: K_TILES=2, N_TILES=1, nv=2 - accumulate-on-write proven by
//       hierarchical peeks: after k-tile 0 the banks hold exactly the
//       partial (tile-0) sums (fresh write), after k-tile 1 exactly
//       partial0+partial1 (added, not overwritten).
//   T2: sequencing monitors across a 2x2 sweep - w_tile_idx sequence
//       0,1,2,3; a_base 0,nv,0,nv; exactly K_TILES*N_TILES inner gos.
//   T3: cfg_err guard - n_vecs > MAX_N_VECS (engine built with
//       MAX_N_VECS=4, nv=5 written), k_tiles=0, and capacity overflow
//       are each rejected: busy never rises, cfg_err sets; a valid go
//       clears it.
//   (T4/T5 - final requant stream + intermediate-not-authoritative -
//    land with the Task 2 drain tap.)
//   T6: the cfg_err legs T3 left untested - n_vecs == 0 (rejected, and
//       specifically NOT read as 2^VEC_W) and n_tiles == 0.
//   T7: the k_tiles*n_vecs > DEPTH leg, on a second MAX_N_VECS=8
//       instance - that leg is masked whenever MAX_N_VECS <= K, which
//       both shipping configs are (see the u_eng_guard header), so it
//       cannot be reached on the main engine or from the tiled top.
//       Includes the kt_x_nv == DEPTH accept case to pin the boundary.
// Golden: 8x4 weight matrix (2 K-tiles of 4x4), 8-element activation
// vectors; y[j] = sum_{i<8} W[i][j]*a[i]. Host layout per the plan's
// data-layout contract: tile (n,k) at index n*k_tiles+k, activation
// chunk k at words k*nv..k*nv+nv-1.
//====================================================================
module tb_ws_tiling_engine;

    localparam int N          = 4;
    localparam int K          = 4;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int OUT_WIDTH  = 8;
    localparam int DEPTH      = 64;
    localparam int VEC_W      = 4;
    localparam int TILE_W     = 4;
    localparam int MAX_NV     = 4;    // deliberately small: T3 exceeds it

    logic clk = 1'b0;
    logic rst_n;

    // outer command
    logic              go;
    logic [VEC_W-1:0]  n_vecs;
    logic [TILE_W-1:0] k_tiles, n_tiles;
    logic              busy, cfg_err;

    // requant config (per-tensor; identity here, checked in T4)
    logic               relu_en;
    logic signed [15:0] scale_mult;
    logic [4:0]         shift_amt;

    // engine <-> core nets
    logic                        core_go, core_busy;
    logic [VEC_W-1:0]            core_n_vecs;
    logic [$clog2(DEPTH/K)-1:0]  w_tile_idx;
    logic [$clog2(DEPTH)-1:0]    a_base;

    // core raw/quant edges
    logic signed [ACC_WIDTH-1:0] sum_out   [N];
    logic                        sum_valid [N];
    logic signed [OUT_WIDTH-1:0] q8_out    [N];
    logic                        q8_valid  [N];

    // buffer write ports (TB is the data loader)
    logic        wr_en_w, wr_en_a;
    logic [5:0]  wr_addr_w, wr_addr_a;
    logic [31:0] wr_data_w, wr_data_a;

    // final-result stream (Task 2)
    logic                        res_valid;
    logic signed [OUT_WIDTH-1:0] res_q8;
    logic [VEC_W-1:0]            res_vec;
    logic [$clog2(N)-1:0]        res_col;
    logic [TILE_W-1:0]           res_ntile;

    logic        perf_clear;
    logic [3:0]  perf_addr;
    logic [31:0] perf_data;

    int errors = 0;

    ws_tiling_engine #(
        .N(N), .K(K), .ACC_WIDTH(ACC_WIDTH), .OUT_WIDTH(OUT_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .TILE_W(TILE_W),
        .MAX_N_VECS(MAX_NV), .SCALE_WIDTH(16), .SHIFT_WIDTH(5)
    ) u_eng (
        .clk          (clk),
        .rst_n        (rst_n),
        .go           (go),
        .n_vecs       (n_vecs),
        .k_tiles      (k_tiles),
        .n_tiles      (n_tiles),
        .busy         (busy),
        .cfg_err      (cfg_err),
        .relu_en      (relu_en),
        .scale_mult   (scale_mult),
        .shift_amt    (shift_amt),
        .core_go      (core_go),
        .core_busy    (core_busy),
        .core_n_vecs  (core_n_vecs),
        .w_tile_idx   (w_tile_idx),
        .a_base       (a_base),
        .sum_in       (sum_out),
        .sum_valid_in (sum_valid),
        .res_valid    (res_valid),
        .res_q8       (res_q8),
        .res_vec      (res_vec),
        .res_col      (res_col),
        .res_ntile    (res_ntile)
    );

    //----------------------------------------------------------------
    // T7 guard-only second instance (no core attached).
    //
    // The kt_x_nv leg (k_tiles*n_vecs > DEPTH) is UNREACHABLE IN
    // ISOLATION whenever MAX_N_VECS <= K. The kt_x_nt guard already
    // caps k_tiles at DEPTH/K, and n_vecs is capped at MAX_N_VECS, so
    // the largest product any accepted-so-far config can present is
    // (DEPTH/K)*MAX_N_VECS. That exceeds DEPTH only when
    // MAX_N_VECS > K. BOTH shipping configurations sit exactly on the
    // masked side of that boundary - this TB's main engine is K=4 /
    // MAX_N_VECS=4, and ws_tpu_tiled_top is K=16 / MAX_N_VECS=16 - so
    // neither can arm the leg, and it cannot be tested from the tiled
    // top at all. This instance is built MAX_N_VECS=8 against K=4
    // solely to reach it. Nothing downstream is wired: a REJECTED go
    // never leaves E_IDLE, and the one accepted go at the end is only
    // observed for busy rising, never run to completion.
    //----------------------------------------------------------------
    logic              g2_go, g2_busy, g2_cfg_err, g2_core_busy;
    logic [VEC_W-1:0]  g2_n_vecs;
    logic [TILE_W-1:0] g2_k_tiles, g2_n_tiles;
    logic signed [ACC_WIDTH-1:0] g2_sum_in    [N];
    logic                        g2_sum_valid [N];
    logic                        g2_core_go;
    logic [VEC_W-1:0]            g2_core_n_vecs;
    logic [$clog2(DEPTH/K)-1:0]  g2_w_tile_idx;
    logic [$clog2(DEPTH)-1:0]    g2_a_base;
    logic                        g2_res_valid;
    logic signed [OUT_WIDTH-1:0] g2_res_q8;
    logic [VEC_W-1:0]            g2_res_vec;
    logic [$clog2(N)-1:0]        g2_res_col;
    logic [TILE_W-1:0]           g2_res_ntile;

    assign g2_core_busy = 1'b0;

    initial begin
        for (int i = 0; i < N; i++) begin
            g2_sum_in[i]    = '0;
            g2_sum_valid[i] = 1'b0;
        end
    end

    ws_tiling_engine #(
        .N(N), .K(K), .ACC_WIDTH(ACC_WIDTH), .OUT_WIDTH(OUT_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .TILE_W(TILE_W),
        .MAX_N_VECS(8), .SCALE_WIDTH(16), .SHIFT_WIDTH(5)
    ) u_eng_guard (
        .clk          (clk),
        .rst_n        (rst_n),
        .go           (g2_go),
        .n_vecs       (g2_n_vecs),
        .k_tiles      (g2_k_tiles),
        .n_tiles      (g2_n_tiles),
        .busy         (g2_busy),
        .cfg_err      (g2_cfg_err),
        .relu_en      (relu_en),
        .scale_mult   (scale_mult),
        .shift_amt    (shift_amt),
        .core_go      (g2_core_go),
        .core_busy    (g2_core_busy),
        .core_n_vecs  (g2_core_n_vecs),
        .w_tile_idx   (g2_w_tile_idx),
        .a_base       (g2_a_base),
        .sum_in       (g2_sum_in),
        .sum_valid_in (g2_sum_valid),
        .res_valid    (g2_res_valid),
        .res_q8       (g2_res_q8),
        .res_vec      (g2_res_vec),
        .res_col      (g2_res_col),
        .res_ntile    (g2_res_ntile)
    );

    // debug read port (unused here - tied off so the ws_unit mux never
    // sees a floating input)
    logic                     dbg_rd_en, dbg_rd_sel;
    logic [$clog2(DEPTH)-1:0] dbg_rd_addr;
    logic [N*DATA_WIDTH-1:0]  dbg_rd_data;
    assign dbg_rd_en   = 1'b0;
    assign dbg_rd_sel  = 1'b0;
    assign dbg_rd_addr = '0;

    ws_quant_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH)
    ) u_ws (
        .dbg_rd_en   (dbg_rd_en),
        .dbg_rd_sel  (dbg_rd_sel),
        .dbg_rd_addr (dbg_rd_addr),
        .dbg_rd_data (dbg_rd_data),
        .clk        (clk),
        .rst_n      (rst_n),
        .go         (core_go),
        .w_tile_idx (w_tile_idx),
        .a_base     (a_base),
        .n_vecs     (core_n_vecs),
        .busy       (core_busy),
        .perf_clear (perf_clear),
        .perf_addr  (perf_addr),
        .perf_data  (perf_data),
        .wr_en_w    (wr_en_w),
        .wr_addr_w  (wr_addr_w),
        .wr_data_w  (wr_data_w),
        .wr_en_a    (wr_en_a),
        .wr_addr_a  (wr_addr_a),
        .wr_data_a  (wr_data_a),
        .sum_out    (sum_out),
        .sum_valid  (sum_valid),
        .q8_out     (q8_out),
        .q8_valid   (q8_valid),
        .relu_en    (relu_en),
        .scale_mult (scale_mult),
        .shift_amt  (shift_amt)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    // watchdog
    initial begin
        #2_000_000;
        $error("[watchdog] TIMEOUT");
        $finish;
    end

    //----------------------------------------------------------------
    // Golden: 8x4 problem for T1 (2 K-tiles x 1 N-tile), extended to
    // 8x8 for T2/T4 (2x2). W[i][gcol] with i<8, gcol<8; tile (n,k)
    // holds resident rows k*4..k*4+3, columns n*4..n*4+3, stored at
    // buffer tile index n*k_tiles + k. av[v][i], i<8.
    //----------------------------------------------------------------
    int W  [8][8];
    int av [2][8];

    function automatic int clamp8(input int v, input int relu);
        int r;
        r = v;
        if (relu != 0 && r < 0) r = 0;
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        return r;
    endfunction

    // partial sum for K-tile k only, global column gcol
    function automatic int partial(input int k, input int v, input int gcol);
        int y;
        y = 0;
        for (int i = 0; i < 4; i++)
            y = y + W[k*4 + i][gcol]*av[v][k*4 + i];
        return y;
    endfunction

    // full K=8 dot product, global column gcol
    function automatic int full_y(input int v, input int gcol);
        int y;
        y = 0;
        for (int i = 0; i < 8; i++)
            y = y + W[i][gcol]*av[v][i];
        return y;
    endfunction

    function automatic [31:0] pack4(input int e0, input int e1,
                                    input int e2, input int e3);
        int b0, b1, b2, b3;
        begin
            b0 = (e0 < 0) ? e0 + 256 : e0;
            b1 = (e1 < 0) ? e1 + 256 : e1;
            b2 = (e2 < 0) ? e2 + 256 : e2;
            b3 = (e3 < 0) ? e3 + 256 : e3;
            pack4 = {b3[7:0], b2[7:0], b1[7:0], b0[7:0]};
        end
    endfunction

    task automatic write_w(input int addr, input [31:0] data);
        @(negedge clk) begin wr_en_w = 1; wr_addr_w = addr; wr_data_w = data; end
        @(negedge clk) wr_en_w = 0;
    endtask

    task automatic write_a(input int addr, input [31:0] data);
        @(negedge clk) begin wr_en_a = 1; wr_addr_a = addr; wr_data_a = data; end
        @(negedge clk) wr_en_a = 0;
    endtask

    // load tile (n,k) for a run with kt K-tiles: buffer index n*kt+k,
    // word r = resident row r = W[k*4+r][n*4 .. n*4+3]
    task automatic load_tile(input int n, input int k, input int kt);
        for (int r = 0; r < 4; r++)
            write_w((n*kt + k)*K + r,
                    pack4(W[k*4+r][n*4+0], W[k*4+r][n*4+1],
                          W[k*4+r][n*4+2], W[k*4+r][n*4+3]));
    endtask

    // load activation chunk k for vector v at word k*nv + v
    task automatic load_avec(input int k, input int v, input int nv);
        write_a(k*nv + v, pack4(av[v][k*4+0], av[v][k*4+1],
                                av[v][k*4+2], av[v][k*4+3]));
    endtask

    //----------------------------------------------------------------
    // Monitors: inner-done counter (core_busy falling edges) and the
    // per-inner-go tile/base recorder (T2's sequencing evidence)
    //----------------------------------------------------------------
    logic core_busy_q = 1'b0;
    int inner_done_cnt = 0;

    always @(posedge clk) begin
        if (rst_n) begin
            if (core_busy_q === 1'b1 && core_busy === 1'b0)
                inner_done_cnt = inner_done_cnt + 1;
            core_busy_q <= core_busy;
        end
    end

    int go_cnt = 0;
    int seen_tidx  [8];
    int seen_abase [8];

    always @(posedge clk) begin
        if (rst_n && core_go === 1'b1) begin
            seen_tidx[go_cnt]  = w_tile_idx;
            seen_abase[go_cnt] = a_base;
            go_cnt = go_cnt + 1;
        end
    end

    //----------------------------------------------------------------
    // Final-result collector (T4) + last-inner-run q8 capture (T5:
    // the intermediate stream requantizes PER-TILE partials and is
    // NOT the final answer - prove it, don't just document it)
    //----------------------------------------------------------------
    int res_cnt = 0;
    int r_val [32];
    int r_v   [32];
    int r_c   [32];
    int r_nt  [32];

    always @(posedge clk) begin
        if (rst_n && res_valid === 1'b1) begin
            r_val[res_cnt] = res_q8;
            r_v[res_cnt]   = res_vec;
            r_c[res_cnt]   = res_col;
            r_nt[res_cnt]  = res_ntile;
            res_cnt = res_cnt + 1;
        end
    end

    int q8_cnt  [4];
    int q8_last [4][4];   // [col][vec] of the most recent inner run

    always @(posedge clk) begin
        if (rst_n) begin
            if (core_go === 1'b1)
                for (int j = 0; j < 4; j++) q8_cnt[j] = 0;
            for (int j = 0; j < 4; j++)
                if (q8_valid[j] === 1'b1) begin
                    q8_last[j][q8_cnt[j]] = q8_out[j];
                    q8_cnt[j] = q8_cnt[j] + 1;
                end
        end
    end

    task automatic check_int(input string name, input int got, input int exp);
        if (got !== exp) begin
            errors++;
            $error("[%s] got %0d, expected %0d", name, got, exp);
        end else begin
            $display("PASS [%s] = %0d", name, got);
        end
    endtask

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    initial begin
        rst_n = 0; go = 0; n_vecs = '0; k_tiles = '0; n_tiles = '0;
        g2_go = 0; g2_n_vecs = '0; g2_k_tiles = '0; g2_n_tiles = '0;
        relu_en = 0; scale_mult = 16'sd1; shift_amt = '0;
        wr_en_w = 0; wr_addr_w = '0; wr_data_w = '0;
        wr_en_a = 0; wr_addr_a = '0; wr_data_a = '0;
        perf_clear = 0; perf_addr = '0;

        // deterministic golden data: mixed signs, partial != full for
        // many elements (T5 relies on that; T1 checks it structurally)
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                W[i][j] = ((i*5 + j*3) % 19) - 9;             // -9..9
        for (int v = 0; v < 2; v++)
            for (int i = 0; i < 8; i++)
                av[v][i] = ((v*11 + i*7) % 15) - 7;           // -7..7

        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        //--- T1: K_TILES=2, N_TILES=1, nv=2 - accumulate-on-write -----
        // layout for kt=2: tiles at idx 0,1; chunks at words 0..1, 2..3
        load_tile(0, 0, 2);
        load_tile(0, 1, 2);
        for (int v = 0; v < 2; v++) begin
            load_avec(0, v, 2);
            load_avec(1, v, 2);
        end

        inner_done_cnt = 0; go_cnt = 0;
        @(negedge clk) begin go = 1; n_vecs = 4'd2; k_tiles = 4'd2; n_tiles = 4'd1; end
        @(negedge clk) go = 0;

        // after K-tile 0 completes (first inner busy fall), the banks
        // must hold exactly the fresh-written partial-0 sums
        while (inner_done_cnt < 1) @(negedge clk);
        repeat (2) @(negedge clk);
        for (int v = 0; v < 2; v++)
            for (int j = 0; j < 4; j++)
                if (u_eng.acc[j][v] !== partial(0, v, j)) begin
                    errors++;
                    $error("[T1 k0] acc[%0d][%0d] = %0d, expected partial0 %0d",
                           j, v, u_eng.acc[j][v], partial(0, v, j));
                end
        $display("PASS [T1 k0] banks hold fresh partial-0 sums");

        // after the sweep, banks must hold partial0+partial1 (added)
        while (busy) @(negedge clk);
        @(negedge clk);
        for (int v = 0; v < 2; v++)
            for (int j = 0; j < 4; j++)
                if (u_eng.acc[j][v] !== full_y(v, j)) begin
                    errors++;
                    $error("[T1 k1] acc[%0d][%0d] = %0d, expected full %0d",
                           j, v, u_eng.acc[j][v], full_y(v, j));
                end
        $display("PASS [T1 k1] banks accumulated, not overwritten");
        check_int("T1 inner gos", go_cnt, 2);

        //--- T2: 2x2 sweep sequencing (tiles 2,3 zero-filled) ---------
        for (int t = 2; t < 4; t++)
            for (int r = 0; r < 4; r++)
                write_w(t*K + r, 32'h0);

        inner_done_cnt = 0; go_cnt = 0;
        @(negedge clk) begin go = 1; n_vecs = 4'd2; k_tiles = 4'd2; n_tiles = 4'd2; end
        @(negedge clk) go = 0;
        while (busy) @(negedge clk);
        @(negedge clk);

        check_int("T2 inner gos", go_cnt, 4);
        check_int("T2 tidx0", seen_tidx[0], 0);
        check_int("T2 tidx1", seen_tidx[1], 1);
        check_int("T2 tidx2", seen_tidx[2], 2);
        check_int("T2 tidx3", seen_tidx[3], 3);
        check_int("T2 abase0", seen_abase[0], 0);
        check_int("T2 abase1", seen_abase[1], 2);
        check_int("T2 abase2", seen_abase[2], 0);
        check_int("T2 abase3", seen_abase[3], 2);

        //--- T3: cfg_err guard - rejects, busy never rises ------------
        // (engine built with MAX_N_VECS=4; nv=5 exceeds it)
        go_cnt = 0;
        @(negedge clk) begin go = 1; n_vecs = 4'd5; k_tiles = 4'd1; n_tiles = 4'd1; end
        @(negedge clk) go = 0;
        repeat (6) @(negedge clk);
        if (busy !== 1'b0 || cfg_err !== 1'b1) begin
            errors++;
            $error("[T3 nv>MAX] busy=%b cfg_err=%b, expected 0/1", busy, cfg_err);
        end else $display("PASS [T3] nv > MAX_N_VECS rejected, cfg_err set");

        @(negedge clk) begin go = 1; n_vecs = 4'd1; k_tiles = 4'd0; n_tiles = 4'd1; end
        @(negedge clk) go = 0;
        repeat (6) @(negedge clk);
        if (busy !== 1'b0 || cfg_err !== 1'b1) begin
            errors++;
            $error("[T3 kt=0] busy=%b cfg_err=%b, expected 0/1", busy, cfg_err);
        end else $display("PASS [T3] k_tiles=0 rejected");

        @(negedge clk) begin go = 1; n_vecs = 4'd1; k_tiles = 4'd8; n_tiles = 4'd8; end
        @(negedge clk) go = 0;
        repeat (6) @(negedge clk);
        if (busy !== 1'b0 || cfg_err !== 1'b1) begin
            errors++;
            $error("[T3 cap] busy=%b cfg_err=%b, expected 0/1", busy, cfg_err);
        end else $display("PASS [T3] tile capacity overflow rejected");

        // a valid go clears cfg_err and runs to completion
        @(negedge clk) begin go = 1; n_vecs = 4'd1; k_tiles = 4'd1; n_tiles = 4'd1; end
        @(negedge clk) go = 0;
        @(negedge clk);
        if (cfg_err !== 1'b0) begin
            errors++; $error("[T3 clear] cfg_err not cleared by accepted go");
        end else $display("PASS [T3] accepted go clears cfg_err");
        while (busy) @(negedge clk);

        //--- T4: full 2x2 golden through the final requant stream -----
        // reload all four REAL tiles (T2 zeroed 2/3)
        for (int n = 0; n < 2; n++)
            for (int k = 0; k < 2; k++)
                load_tile(n, k, 2);

        res_cnt = 0;
        @(negedge clk) begin go = 1; n_vecs = 4'd2; k_tiles = 4'd2; n_tiles = 4'd2; end
        @(negedge clk) go = 0;
        while (busy) @(negedge clk);
        repeat (2) @(negedge clk);

        check_int("T4 res count", res_cnt, 16);
        begin
            int idx;
            idx = 0;
            for (int nt = 0; nt < 2; nt++)
                for (int v = 0; v < 2; v++)
                    for (int j = 0; j < 4; j++) begin
                        if (r_nt[idx] !== nt || r_v[idx] !== v || r_c[idx] !== j) begin
                            errors++;
                            $error("[T4 order] idx %0d = (nt%0d v%0d c%0d), expected (nt%0d v%0d c%0d)",
                                   idx, r_nt[idx], r_v[idx], r_c[idx], nt, v, j);
                        end
                        if (r_val[idx] !== clamp8(full_y(v, nt*4 + j), 0)) begin
                            errors++;
                            $error("[T4 value] res(nt%0d v%0d c%0d) = %0d, expected %0d",
                                   nt, v, j, r_val[idx], clamp8(full_y(v, nt*4 + j), 0));
                        end
                        idx = idx + 1;
                    end
        end
        $display("PASS [T4] 2x2 sweep: 16 final results, order + values exact");

        //--- T5: the intermediate q8 stream is NOT the final answer ---
        // last inner run was tile (n=1, k=1): its q8 = requant of the
        // k=1 PARTIAL alone. Golden guarantees at least one element
        // where clamp8(partial) != clamp8(full); check the hardware
        // agrees on both sides of that difference.
        begin
            int diffs;
            diffs = 0;
            for (int v = 0; v < 2; v++)
                for (int j = 0; j < 4; j++) begin
                    if (q8_last[j][v] !== clamp8(partial(1, v, 4 + j), 0)) begin
                        errors++;
                        $error("[T5] intermediate q8[%0d][%0d] = %0d, expected per-tile partial %0d",
                               j, v, q8_last[j][v], clamp8(partial(1, v, 4 + j), 0));
                    end
                    if (clamp8(partial(1, v, 4 + j), 0) != clamp8(full_y(v, 4 + j), 0))
                        diffs = diffs + 1;
                end
            if (diffs == 0) begin
                errors++;
                $error("[T5] golden degenerate: no element distinguishes partial from full");
            end else begin
                $display("PASS [T5] intermediate stream = per-tile partials (%0d elements differ from final; res_* is the only authoritative output)", diffs);
            end
        end

        //--- T6: the two cfg_err legs T3 left untested -----------------
        // T3 covers n_vecs > MAX_N_VECS, k_tiles == 0 and the kt_x_nt
        // capacity leg. n_vecs == 0 and n_tiles == 0 were never
        // exercised. n_vecs == 0 matters most: the engine header calls
        // out that it is REJECTED and does not silently mean 2^VEC_W.
        // T4/T5 ran accepted gos, so cfg_err must be clear on entry -
        // assert that, or a stale sticky bit would fake both passes.
        if (cfg_err !== 1'b0) begin
            errors++;
            $error("[T6] cfg_err = %b on entry, expected 0 (T4/T5 gos were accepted)", cfg_err);
        end

        @(negedge clk) begin go = 1; n_vecs = 4'd0; k_tiles = 4'd1; n_tiles = 4'd1; end
        @(negedge clk) go = 0;
        repeat (6) @(negedge clk);
        if (busy !== 1'b0 || cfg_err !== 1'b1) begin
            errors++;
            $error("[T6 nv=0] busy=%b cfg_err=%b, expected 0/1", busy, cfg_err);
        end else $display("PASS [T6] n_vecs=0 rejected (does not mean 2^VEC_W)");

        @(negedge clk) begin go = 1; n_vecs = 4'd1; k_tiles = 4'd1; n_tiles = 4'd0; end
        @(negedge clk) go = 0;
        repeat (6) @(negedge clk);
        if (busy !== 1'b0 || cfg_err !== 1'b1) begin
            errors++;
            $error("[T6 nt=0] busy=%b cfg_err=%b, expected 0/1", busy, cfg_err);
        end else $display("PASS [T6] n_tiles=0 rejected");

        //--- T7: the kt_x_nv leg, on the MAX_N_VECS=8 instance --------
        // Unreachable on the main engine (MAX_N_VECS == K == 4); see
        // the u_eng_guard header. Here DEPTH/K = 16 so kt_x_nt allows
        // k_tiles up to 16, and MAX_N_VECS = 8 > K = 4 opens the gap.
        //   reject: nv=8, kt=9, nt=1 -> kt_x_nv = 72 > DEPTH(64),
        //           while kt_x_nt = 9 <= 16 and nv = 8 <= MAX_N_VECS,
        //           so kt_x_nv is the ONLY leg that can be firing.
        if (g2_cfg_err !== 1'b0 || g2_busy !== 1'b0) begin
            errors++;
            $error("[T7] guard instance not idle/clear on entry: busy=%b cfg_err=%b",
                   g2_busy, g2_cfg_err);
        end

        @(negedge clk) begin g2_go = 1; g2_n_vecs = 4'd8; g2_k_tiles = 4'd9; g2_n_tiles = 4'd1; end
        @(negedge clk) g2_go = 0;
        repeat (6) @(negedge clk);
        if (g2_busy !== 1'b0 || g2_cfg_err !== 1'b1) begin
            errors++;
            $error("[T7 kt_x_nv] busy=%b cfg_err=%b, expected 0/1", g2_busy, g2_cfg_err);
        end else $display("PASS [T7] k_tiles*n_vecs > DEPTH rejected (leg isolated)");

        //   accept: nv=8, kt=8 -> kt_x_nv = 64, NOT > DEPTH. Same shape,
        //   one tile less, so this pins the boundary at exactly DEPTH
        //   and proves the rejection above came from kt_x_nv alone.
        //   Left running deliberately - no core is attached, so it
        //   parks in E_WAITB; only the busy rise is observed.
        @(negedge clk) begin g2_go = 1; g2_n_vecs = 4'd8; g2_k_tiles = 4'd8; g2_n_tiles = 4'd1; end
        @(negedge clk) g2_go = 0;
        repeat (6) @(negedge clk);
        if (g2_busy !== 1'b1 || g2_cfg_err !== 1'b0) begin
            errors++;
            $error("[T7 boundary] kt_x_nv == DEPTH should be accepted: busy=%b cfg_err=%b, expected 1/0",
                   g2_busy, g2_cfg_err);
        end else $display("PASS [T7] k_tiles*n_vecs == DEPTH accepted (boundary is exact)");

        //--- summary ---------------------------------------------------
        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
