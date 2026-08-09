`timescale 1ns / 1ps
//====================================================================
// tb_ws_quant_unit: quantized WS matvec through ws_quant_unit
// (ws_unit untouched + requant lane per column on the live stream).
//   Identity quant (scale=1, shift=0), so golden INT8 = clamp
//   (+ optional ReLU) of the matvec int golden - wiring and cadence
//   are under test; scale/round math is proven by tb_requant.
//   T1: 1 vector, relu off - values + cadence (max 1 column
//       valid/cycle at n_vecs=1)
//   T2: 4 vectors back-to-back, relu off - 4 in-order results per
//       column, and THE SPEC 3b CADENCE MEASUREMENT OF RECORD:
//       peak concurrent valid columns must equal min(n_vecs,N)=4,
//       certifying that the one-value-per-cycle assumption fails
//       and the lane-per-column structure is required.
//   T3: 1 vector, relu on - negatives zeroed at the int8 output.
//====================================================================
module tb_ws_quant_unit;

    localparam int N          = 4;
    localparam int K          = 4;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int OUT_WIDTH  = 8;
    localparam int DEPTH      = 64;
    localparam int VEC_W      = 4;

    // DUT I/O
    logic clk = 1'b0;
    logic rst_n, go;
    logic [$clog2(DEPTH/K)-1:0] w_tile_idx;
    logic [$clog2(DEPTH)-1:0]   a_base;
    logic [VEC_W-1:0]           n_vecs;
    logic                       busy;
    logic                       wr_en_w, wr_en_a;
    logic [$clog2(DEPTH)-1:0]   wr_addr_w, wr_addr_a;
    logic [31:0]                wr_data_w, wr_data_a;
    logic signed [ACC_WIDTH-1:0] sum_out [N];
    logic                        sum_valid [N];
    logic signed [OUT_WIDTH-1:0] q8_out [N];
    logic                        q8_valid [N];
    logic                        relu_en;
    logic signed [15:0]          scale_mult;
    logic [4:0]                  shift_amt;
    logic        perf_clear;
    logic [3:0]  perf_addr;
    logic [31:0] perf_data;

    int errors = 0;

    // debug read port (unused here - tied off so the ws_unit mux never
    // sees a floating input)
    logic                     dbg_rd_en, dbg_rd_sel;
    logic [$clog2(DEPTH)-1:0] dbg_rd_addr;
    logic [N*DATA_WIDTH-1:0]  dbg_rd_data;
    assign dbg_rd_en   = 1'b0;
    assign dbg_rd_sel  = 1'b0;
    assign dbg_rd_addr = '0;

    //----------------------------------------------------------------
    // D-CHECK: the new done pulse, delayed by RQ_PIPE inside the DUT so
    // it lands on the final q8_valid rather than on the controller's
    // last sum_valid. Free-running, so every run here is covered.
    //----------------------------------------------------------------
    // ANY lane, not lane 0. The array is column-skewed, so column 0
    // drains FIRST and column N-1 last - a done that fires up to
    // RQ_PIPE cycles early still lands after lane 0's final beat and
    // is invisible if only lane 0 is watched. That is not hypothetical:
    // the first version of this monitor did watch lane 0, and the
    // control that removes the RQ_PIPE delay passed against it.
    logic dut_done, any_q8;
    always_comb begin
        any_q8 = 1'b0;
        for (int j = 0; j < N; j++) if (q8_valid[j]) any_q8 = 1'b1;
    end

    // "after done" must be LATCHED. Counting `!busy && !done` instead
    // counts every beat in the post-busy drain, including ones that
    // arrive BEFORE done - which is not the property, and made a
    // correctly-placed done look broken.
    // THE GO-CLEAR MUST WIN when go and done land on the same cycle,
    // and they do: this testbench issues the next go in the very cycle
    // the previous done fires. With the clear written first, dk_seen
    // stayed set across the boundary and the NEXT run's beats were all
    // counted as "after done" - 11 of them, which read as an RTL bug
    // when the RTL was right. A cycle probe settled it: ctl_done 36,
    // last q8 beat 37, done 38. Last assignment wins, so the clear
    // goes last.
    int  dk_go, dk_done, dk_at, dk_after, dk_busy;
    logic dk_seen;
    always @(posedge clk) if (rst_n) begin
        if (any_q8 && dk_seen && !dut_done) dk_after <= dk_after + 1;
        if (dut_done) begin
            dk_done <= dk_done + 1;
            dk_seen <= 1'b1;
            if (any_q8) dk_at   <= dk_at + 1;
            if (busy)   dk_busy <= dk_busy + 1;
        end
        if (go && !busy) begin
            dk_go   <= dk_go + 1;
            dk_seen <= 1'b0;
        end
    end
    initial begin dk_go = 0; dk_done = 0; dk_at = 0; dk_after = 0; dk_busy = 0; dk_seen = 1'b0; end

    ws_quant_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH)
    ) dut (
        .dbg_rd_en   (dbg_rd_en),
        .dbg_rd_sel  (dbg_rd_sel),
        .dbg_rd_addr (dbg_rd_addr),
        .dbg_rd_data (dbg_rd_data),
        .clk        (clk),
        .rst_n      (rst_n),
        .go         (go),
        .w_tile_idx (w_tile_idx),
        .a_base     (a_base),
        .n_vecs     (n_vecs),
        .busy       (busy),
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
        .done       (dut_done),
        .relu_en    (relu_en),
        .scale_mult (scale_mult),
        .shift_amt  (shift_amt)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    //----------------------------------------------------------------
    // Golden data: magnitudes chosen so some outputs clip both ways
    // and some pass unclipped (products up to ~9*8*4 = 288)
    //----------------------------------------------------------------
    int w  [4][4];    // resident weights w[i][j]
    int av [4][4];    // av[v][i]: activation vector v
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

    //----------------------------------------------------------------
    // Pack 4 int8 values into one word (element 0 in [7:0]); manual
    // two's complement
    //----------------------------------------------------------------
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

    //----------------------------------------------------------------
    // Collectors + THE CADENCE METER (spec 3b): per-cycle count of
    // simultaneously-valid quantized columns, peak tracked per run
    //----------------------------------------------------------------
    int got     [4][4];
    int got_cnt [4];
    int cad, cad_max;

    always @(posedge clk) begin
        if (rst_n) begin
            cad = 0;
            for (int j = 0; j < N; j++) begin
                if (q8_valid[j] === 1'b1) begin
                    got[j][got_cnt[j]] = q8_out[j];
                    got_cnt[j] = got_cnt[j] + 1;
                    cad = cad + 1;
                end
            end
            if (cad > cad_max) cad_max = cad;
        end
    end

    //----------------------------------------------------------------
    // Run one tile and check every column/vector + the cadence peak
    //----------------------------------------------------------------
    task automatic run_and_check(input int tile, input int base,
                                 input int nv, input int relu,
                                 input string name);
        for (int j = 0; j < N; j++) got_cnt[j] = 0;
        cad_max = 0;

        @(negedge clk) begin
            go = 1; w_tile_idx = tile; a_base = base;
            n_vecs = VEC_W'(nv); relu_en = relu;
        end
        @(negedge clk) go = 0;
        while (busy) @(negedge clk);
        @(negedge clk);

        for (int j = 0; j < N; j++) begin
            if (got_cnt[j] !== nv) begin
                errors++;
                $error("[%s] column %0d: %0d results, expected %0d", name, j, got_cnt[j], nv);
            end else begin
                for (int v = 0; v < nv; v++) begin
                    y_exp = clamp8(matvec(v, j), relu);
                    if (got[j][v] !== y_exp) begin
                        errors++;
                        $error("[%s] q8 y%0d[%0d] = %0d, expected %0d (int32 %0d)",
                               name, v, j, got[j][v], y_exp, matvec(v, j));
                    end
                end
            end
        end
        $display("PASS [%s] (checked %0d columns x %0d vectors)", name, N, nv);

        // cadence certification: peak concurrent columns = min(nv, N)
        if (cad_max !== ((nv < N) ? nv : N)) begin
            errors++;
            $error("[%s cadence] peak concurrent valid columns = %0d, predicted min(n_vecs,N) = %0d",
                   name, cad_max, (nv < N) ? nv : N);
        end else begin
            $display("PASS [%s cadence] peak concurrent columns = %0d", name, cad_max);
        end
    endtask

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    initial begin
        rst_n = 0; go = 0; w_tile_idx = '0; a_base = '0; n_vecs = '0;
        wr_en_w = 0; wr_addr_w = '0; wr_data_w = '0;
        wr_en_a = 0; wr_addr_a = '0; wr_data_a = '0;
        relu_en = 0; scale_mult = 16'sd1; shift_amt = '0;   // identity quant
        perf_clear = 0; perf_addr = '0;

        // weights: row magnitudes force clipping in some columns,
        // row 2 small so some outputs stay in range
        w[0][0]=9;  w[0][1]=-8; w[0][2]=2;  w[0][3]=-9;
        w[1][0]=-7; w[1][1]=9;  w[1][2]=-1; w[1][3]=8;
        w[2][0]=1;  w[2][1]=-2; w[2][2]=3;  w[2][3]=-1;
        w[3][0]=8;  w[3][1]=-9; w[3][2]=1;  w[3][3]=9;
        // four activation vectors (used 1 or 4 at a time)
        av[0][0]=7;  av[0][1]=-6; av[0][2]=5;  av[0][3]=8;
        av[1][0]=-8; av[1][1]=7;  av[1][2]=-2; av[1][3]=-5;
        av[2][0]=1;  av[2][1]=2;  av[2][2]=-3; av[2][3]=0;
        av[3][0]=-4; av[3][1]=-7; av[3][2]=6;  av[3][3]=3;

        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // load: weight tile at idx 1, vectors at words 8..11
        for (int i = 0; i < 4; i++)
            write_w(1*K + i, pack4(w[i][0], w[i][1], w[i][2], w[i][3]));
        for (int v = 0; v < 4; v++)
            write_a(8 + v, pack4(av[v][0], av[v][1], av[v][2], av[v][3]));

        //--- T1: single vector, relu off ------------------------------
        run_and_check(1, 8, 1, 0, "T1 1-vec identity quant");

        //--- T2: 4 vectors back-to-back - cadence measurement ---------
        run_and_check(1, 8, 4, 0, "T2 4-vec stream");

        //--- T3: single vector, relu on -------------------------------
        run_and_check(1, 8, 1, 1, "T3 1-vec relu");

        //--- summary ---------------------------------------------------
        repeat (2) @(negedge clk);
        // ---- D-CHECK -----------------------------------------------
        if (dk_done != dk_go) begin
            errors++;
            $error("[D] done fired %0d times for %0d go(s)", dk_done, dk_go);
        end
        if (dk_after != 0) begin
            errors++;
            $error("[D] %0d q8 beat(s) arrived AFTER done", dk_after);
        end
        if (dk_busy != 0) begin
            errors++;
            $error("[D] done asserted %0d time(s) while still busy", dk_busy);
        end
        if (dk_done == dk_go && dk_after == 0 && dk_busy == 0)
            $display("PASS [D] %0d done pulses, none early, none while busy", dk_done);

        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
