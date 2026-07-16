`timescale 1ns / 1ps
//====================================================================
// tb_axi_lite_ctrl v3 (spec step 4): full integration - axi_lite_ctrl
// driving a REAL ws_quant_unit (N=4) across genuinely asynchronous
// clocks (aclk 100 MHz fixed; cclk ~263 MHz average with random
// per-half-cycle jitter, so the phase relationship sweeps through
// all alignments across the run - spec section 5).
//   P1: STATUS + unmapped addresses read 0 after reset
//   P2: staggered AW-first / W-first writes (ordering), no stray go
//   Functional suite, looped 3x for phase coverage:
//     F1: 15-vector identity-quant run. STATUS.busy observed mid-run,
//         done polled, exactly one core_go pulse, all 4x15 q8 results
//         exact vs golden, perf load/stream/flush/tile_swaps exact
//         via PERF_SEL/PERF_DATA (cumulative expectations).
//     F2: relu run - REQUANT_CFG provably reaches the requant lanes
//         (negative int32 results clamp to 0 at the int8 edge).
//     F3: back-to-back N_VECS writes - the core-side value is only
//         ever old/first/final (never torn), converges on the final
//         value, and a run then streams exactly that many vectors.
//   P3: held-RREADY read at the end - rvalid/rdata stable.
// Single-tile scope: w_tile_idx/a_base tied 0 (no registers in the
// spec map). Buffer preload uses the raw write ports on the core
// clock - data movement is the future DMA/AXI-HP path, not AXI-Lite.
//====================================================================
module tb_axi_lite_ctrl;

    localparam int N          = 4;
    localparam int K          = 4;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int OUT_WIDTH  = 8;
    localparam int DEPTH      = 64;
    localparam int VEC_W      = 4;

    localparam [5:0] ADDR_CTRL   = 6'h00;
    localparam [5:0] ADDR_STATUS = 6'h04;
    localparam [5:0] ADDR_NVECS  = 6'h08;
    localparam [5:0] ADDR_CFG    = 6'h0C;
    localparam [5:0] ADDR_PSEL   = 6'h10;
    localparam [5:0] ADDR_PDATA  = 6'h14;

    localparam int POLL_MAX = 50;
    // crossing-settle waits (generous vs the ~6 aclk + ~6 cclk actual)
    localparam int CFG_SETTLE_ACLKS  = 20;
    localparam int PERF_SETTLE_ACLKS = 20;

    //----------------------------------------------------------------
    // Clocks + resets (two genuinely asynchronous domains)
    //----------------------------------------------------------------
    logic aclk = 1'b0;
    logic cclk = 1'b0;
    logic aresetn, crst_n;

    always #5 aclk = ~aclk;    // 100 MHz AXI clock

    initial begin
        forever begin
            #(1.7 + $urandom_range(0, 400)/1000.0) cclk = ~cclk;
        end
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
    // Core-side nets (axi_lite_ctrl <-> ws_quant_unit) + WS extras
    //----------------------------------------------------------------
    logic                        core_go, core_busy;
    logic [VEC_W-1:0]            core_n_vecs;
    logic                        core_relu_en;
    logic signed [15:0]          core_scale_mult;
    logic [4:0]                  core_shift_amt;
    logic [3:0]                  core_perf_addr;
    logic [31:0]                 core_perf_data;

    logic [3:0]                  w_tile_idx;   // tied 0: single-tile scope
    logic [5:0]                  a_base;       // tied 0
    logic                        perf_clear;   // tied 0 (no AXI clear bit)
    logic                        wr_en_w, wr_en_a;
    logic [5:0]                  wr_addr_w, wr_addr_a;
    logic [31:0]                 wr_data_w, wr_data_a;
    logic signed [ACC_WIDTH-1:0] sum_out   [N];
    logic                        sum_valid [N];
    logic signed [OUT_WIDTH-1:0] q8_out    [N];
    logic                        q8_valid  [N];

    int errors  = 0;
    int go_seen = 0;
    int go_base = 0;

    axi_lite_ctrl #(
        .VEC_W(VEC_W), .SCALE_WIDTH(16), .SHIFT_WIDTH(5)
    ) dut (
        .s_axi_aclk      (aclk),
        .s_axi_aresetn   (aresetn),
        .s_axi_awvalid   (awvalid),
        .s_axi_awready   (awready),
        .s_axi_awaddr    (awaddr),
        .s_axi_wvalid    (wvalid),
        .s_axi_wready    (wready),
        .s_axi_wdata     (wdata),
        .s_axi_wstrb     (wstrb),
        .s_axi_bvalid    (bvalid),
        .s_axi_bready    (bready),
        .s_axi_bresp     (bresp),
        .s_axi_arvalid   (arvalid),
        .s_axi_arready   (arready),
        .s_axi_araddr    (araddr),
        .s_axi_rvalid    (rvalid),
        .s_axi_rready    (rready),
        .s_axi_rdata     (rdata),
        .s_axi_rresp     (rresp),
        .core_clk        (cclk),
        .core_rst_n      (crst_n),
        .core_go         (core_go),
        .core_busy       (core_busy),
        .core_n_vecs     (core_n_vecs),
        .core_relu_en    (core_relu_en),
        .core_scale_mult (core_scale_mult),
        .core_shift_amt  (core_shift_amt),
        .core_perf_addr  (core_perf_addr),
        .core_perf_data  (core_perf_data)
    );

    ws_quant_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH)
    ) u_ws (
        .clk        (cclk),
        .rst_n      (crst_n),
        .go         (core_go),
        .w_tile_idx (w_tile_idx),
        .a_base     (a_base),
        .n_vecs     (core_n_vecs),
        .busy       (core_busy),
        .perf_clear (perf_clear),
        .perf_addr  (core_perf_addr),
        .perf_data  (core_perf_data),
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
        .relu_en    (core_relu_en),
        .scale_mult (core_scale_mult),
        .shift_amt  (core_shift_amt)
    );

    //----------------------------------------------------------------
    // Core-domain monitors: go pulse counter, q8 collectors, and the
    // torn-value watch on core_n_vecs (spec section 5's back-to-back
    // write check: never an intermediate, neither-old-nor-new value)
    //----------------------------------------------------------------
    always @(posedge cclk)
        if (crst_n && core_go === 1'b1) go_seen = go_seen + 1;

    int got     [4][16];
    int got_cnt [4];

    always @(posedge cclk) begin
        if (crst_n) begin
            for (int j = 0; j < N; j++) begin
                if (q8_valid[j] === 1'b1) begin
                    got[j][got_cnt[j]] = q8_out[j];
                    got_cnt[j] = got_cnt[j] + 1;
                end
            end
        end
    end

    logic [3:0] nv_ok0, nv_ok1, nv_ok2;
    int nv_mon_en = 0;
    int nv_viol   = 0;

    always @(posedge cclk) begin
        if (nv_mon_en != 0) begin
            if (core_n_vecs !== nv_ok0 && core_n_vecs !== nv_ok1
                && core_n_vecs !== nv_ok2) begin
                nv_viol = nv_viol + 1;
                $error("[F3] torn N_VECS at core: %0d", core_n_vecs);
            end
        end
    end

    //----------------------------------------------------------------
    // Golden model (int source of truth; manual two's complement)
    //----------------------------------------------------------------
    int w  [4][4];     // resident weights w[i][j]
    int av [16][4];    // av[v][i]: activation vector v
    int y_exp;

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
        for (int i = 0; i < 4; i++)
            y = y + w[i][j]*av[v][i];
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

    // buffer preload (core clock - the future DMA path, not AXI-Lite)
    task automatic write_w(input int addr, input [31:0] data);
        @(negedge cclk) begin wr_en_w = 1; wr_addr_w = addr; wr_data_w = data; end
        @(negedge cclk) wr_en_w = 0;
    endtask

    task automatic write_a(input int addr, input [31:0] data);
        @(negedge cclk) begin wr_en_a = 1; wr_addr_a = addr; wr_data_a = data; end
        @(negedge cclk) wr_en_a = 0;
    endtask

    //----------------------------------------------------------------
    // AXI-Lite BFM (drive negedge, sample posedge) - identical to the
    // Task 2/3 version
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

    task automatic axi_write_aw_first(input [5:0] addr, input [31:0] data);
        @(negedge aclk) begin awvalid = 1; awaddr = addr; bready = 1; end
        do @(posedge aclk); while (awready !== 1'b1);
        @(negedge aclk) awvalid = 0;
        repeat (2) @(negedge aclk);
        wvalid = 1; wdata = data;
        do @(posedge aclk); while (wready !== 1'b1);
        @(negedge aclk) wvalid = 0;
        do @(posedge aclk); while (bvalid !== 1'b1);
        if (bresp !== 2'b00) begin
            errors++; $error("aw_first: bresp = %b", bresp);
        end
        @(negedge aclk) bready = 0;
    endtask

    task automatic axi_write_w_first(input [5:0] addr, input [31:0] data);
        @(negedge aclk) begin wvalid = 1; wdata = data; bready = 1; end
        do @(posedge aclk); while (wready !== 1'b1);
        @(negedge aclk) wvalid = 0;
        repeat (2) @(negedge aclk);
        awvalid = 1; awaddr = addr;
        do @(posedge aclk); while (awready !== 1'b1);
        @(negedge aclk) awvalid = 0;
        do @(posedge aclk); while (bvalid !== 1'b1);
        if (bresp !== 2'b00) begin
            errors++; $error("w_first: bresp = %b", bresp);
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

    task automatic axi_read_slow(input [5:0] addr, output [31:0] data);
        logic [31:0] first;
        @(negedge aclk) begin arvalid = 1; araddr = addr; rready = 0; end
        do @(posedge aclk); while (arready !== 1'b1);
        @(negedge aclk) arvalid = 0;
        do @(posedge aclk); while (rvalid !== 1'b1);
        first = rdata;
        repeat (3) @(posedge aclk);
        if (rvalid !== 1'b1 || rdata !== first) begin
            errors++;
            $error("axi_read_slow: rvalid/rdata not held (rvalid=%b)", rvalid);
        end
        @(negedge aclk) rready = 1;
        @(posedge aclk);
        data = rdata;
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
    // Perf readback over AXI (PERF_SEL write, bounded wait, PERF_DATA
    // read). Expectations are CUMULATIVE since reset - budgets from
    // tb_ws_controller T4: load=N+2, stream=nv+1, flush=2N+1, 1 swap
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
    // One full run through the register file: program N_VECS +
    // REQUANT_CFG (identity scale, optional relu), go, busy/done via
    // STATUS, values vs golden, perf counters vs cumulative budget
    //----------------------------------------------------------------
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
            // busy_axi is visible ~4 aclk after the go commit; a
            // 15-vec run holds busy ~31 cclk (~118 ns = ~12 aclk):
            // sampling ~7-9 aclk after the commit lands mid-window
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
                    y_exp = clamp8(matvec(v, j), relu);
                    if (got[j][v] !== y_exp) begin
                        errors++;
                        $error("[%s] q8 y%0d[%0d] = %0d, expected %0d",
                               name, v, j, got[j][v], y_exp);
                    end
                end
            end
        end
        $display("PASS [%s] %0d vectors x %0d columns checked", name, nv, N);

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
        w_tile_idx = '0; a_base = '0; perf_clear = 0;
        wr_en_w = 0; wr_addr_w = '0; wr_data_w = '0;
        wr_en_a = 0; wr_addr_a = '0; wr_data_a = '0;
        nv_ok0 = '0; nv_ok1 = '0; nv_ok2 = '0;

        // weights clip both ways at identity quant; 16 deterministic
        // activation vectors spread over [-9, 9]
        w[0][0]=9;  w[0][1]=-8; w[0][2]=2;  w[0][3]=-9;
        w[1][0]=-7; w[1][1]=9;  w[1][2]=-1; w[1][3]=8;
        w[2][0]=1;  w[2][1]=-2; w[2][2]=3;  w[2][3]=-1;
        w[3][0]=8;  w[3][1]=-9; w[3][2]=1;  w[3][3]=9;
        for (int v = 0; v < 16; v++)
            for (int i = 0; i < 4; i++)
                av[v][i] = ((v*7 + i*3) % 19) - 9;

        repeat (3) @(negedge aclk);
        aresetn = 1;
        crst_n  = 1;
        repeat (2) @(negedge aclk);

        // preload weight tile 0 (words 0..3) + 16 vectors (words 0..15)
        for (int i = 0; i < 4; i++)
            write_w(i, pack4(w[i][0], w[i][1], w[i][2], w[i][3]));
        for (int v = 0; v < 16; v++)
            write_a(v, pack4(av[v][0], av[v][1], av[v][2], av[v][3]));

        //--- P1: reset state ------------------------------------------
        axi_read(ADDR_STATUS, rd);
        if (rd !== 32'h0) begin
            errors++; $error("[P1] STATUS after reset = %h, expected 0", rd);
        end else $display("PASS [P1] STATUS reads 0 after reset");
        axi_read(6'h18, rd);
        if (rd !== 32'h0) begin
            errors++; $error("[P1] unmapped 0x18 = %h, expected 0", rd);
        end else $display("PASS [P1] unmapped address reads 0");

        //--- P2: staggered channel ordering, no stray go ---------------
        go_base = go_seen;
        axi_write_aw_first(ADDR_CTRL, 32'h0);
        axi_write_w_first(ADDR_CTRL, 32'h0);
        repeat (10) @(negedge aclk);
        if (go_seen - go_base !== 0) begin
            errors++;
            $error("[P2] stray go pulse (delta=%0d)", go_seen - go_base);
        end else $display("PASS [P2] staggered AW/W writes, no stray go");

        //--- functional suite, 3 iterations for async phase coverage ---
        for (int it = 0; it < 3; it++) begin
            $display("--- async iteration %0d ---", it);

            run_via_axi(15, 0, 1, "F1 identity 15-vec");
            run_via_axi(1, 1, 0, "F2 relu 1-vec");

            //--- F3: back-to-back N_VECS writes, torn-value watch ------
            // second write lands while the first handshake is often
            // still in flight (round trip ~4-5 aclk vs ~5-6 aclk
            // between commits; jitter + 3 iterations cover both cases)
            nv_ok0 = 4'd1;      // current core value (F2's nv)
            nv_ok1 = 4'd2;      // first write
            nv_ok2 = 4'd5;      // final write
            nv_viol = 0;
            nv_mon_en = 1;
            axi_write(ADDR_NVECS, 32'd2);
            axi_write(ADDR_NVECS, 32'd5);
            repeat (20) @(negedge aclk);
            nv_mon_en = 0;
            errors = errors + nv_viol;
            if (core_n_vecs !== 4'd5) begin
                errors++;
                $error("[F3] core n_vecs = %0d, expected final 5", core_n_vecs);
            end
            if (nv_viol == 0 && core_n_vecs === 4'd5)
                $display("PASS [F3] b2b N_VECS: never torn, converged on 5");
            run_via_axi(5, 0, 0, "F3 stream n_vecs=5");
        end

        //--- P3: held-RREADY read (done still set from the last run) ---
        axi_read_slow(ADDR_STATUS, rd);
        if (rd[1] !== 1'b1) begin
            errors++; $error("[P3] STATUS.done lost during held read");
        end else $display("PASS [P3] rvalid/rdata held until rready");

        //--- summary ----------------------------------------------------
        repeat (2) @(negedge aclk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
