`timescale 1ns / 1ps
//====================================================================
// tb_ws_controller: golden-model check of ws_unit (controller +
// buffers + skew + array), single-tile matrix-vector.
//   T1: tile at w_tile_idx=2, one activation vector at a_base=7 -
//       every column output collected on its own sum_valid and
//       compared against y[j] = sum_i w[i][j]*a[i] (int golden,
//       manual two's complement packing, no $signed)
//   T2: second go on a different tile/vector - clean re-load, no
//       corruption from tile 1
//   T3: back-to-back tiles (spec step 4, revised): second go issued
//       on the FIRST idle cycle after the first tile - per-column
//       results of both tiles collected in order, total busy
//       exactly 2x17 with a 1-cycle idle gap. Certifies the swap
//       cost: FLUSH 9 + LOAD 6 = 15 busy cycles per swap, with the
//       BRAM fetch already folded into the LOAD shift (the staging
//       double-buffer was dropped as obsolete-by-construction; see
//       docs/ws_design_notes.md).
//   Timing certification (written-down predictions, spec 3.3/8):
//       LOAD = N+2 = 6, STREAM = n_vecs+1 = 2, FLUSH = 2N+1 = 9
//       => busy exactly 17 cycles per 1-vector tile, and every
//       result arrives while busy is still high.
//====================================================================
module tb_ws_controller;

    localparam int N          = 4;
    localparam int K          = 4;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int DEPTH      = 64;
    localparam int VEC_W      = 4;

    localparam int PRED_BUSY_1VEC = (N+2) + 2 + (2*N+1);   // 17

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

    int errors = 0;

    ws_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .go         (go),
        .w_tile_idx (w_tile_idx),
        .a_base     (a_base),
        .n_vecs     (n_vecs),
        .busy       (busy),
        .wr_en_w    (wr_en_w),
        .wr_addr_w  (wr_addr_w),
        .wr_data_w  (wr_data_w),
        .wr_en_a    (wr_en_a),
        .wr_addr_a  (wr_addr_a),
        .wr_data_a  (wr_data_a),
        .sum_out    (sum_out),
        .sum_valid  (sum_valid)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    //----------------------------------------------------------------
    // Golden data (int, source of truth; bytes derived from these)
    //----------------------------------------------------------------
    int w1 [4][4];   // resident weights, tile idx 2: w1[i][j]
    int a1 [4];      // activation vector 1
    int w2 [4][4];   // tile idx 0
    int a2 [4];
    int y_exp  [4];  // per-test expected column outputs
    int y1_exp [4];  // T3: expected for tile A
    int y2_exp [4];  // T3: expected for tile B

    //----------------------------------------------------------------
    // Pack 4 int8 values into one buffer word, element 0 in [7:0].
    // Manual two's complement - no $signed (known Icarus issue).
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

    //----------------------------------------------------------------
    // Buffer write helpers (drive negedge)
    //----------------------------------------------------------------
    task automatic write_w(input int addr, input [31:0] data);
        @(negedge clk) begin wr_en_w = 1; wr_addr_w = addr; wr_data_w = data; end
        @(negedge clk) wr_en_w = 0;
    endtask

    task automatic write_a(input int addr, input [31:0] data);
        @(negedge clk) begin wr_en_a = 1; wr_addr_a = addr; wr_data_a = data; end
        @(negedge clk) wr_en_a = 0;
    endtask

    //----------------------------------------------------------------
    // Result collection: every sum_valid cycle appends that column's
    // sum_out. Busy-cycle counter certifies the cycle-budget
    // predictions. late_result trips if a valid appears after busy.
    //----------------------------------------------------------------
    int got     [4][4];
    int got_cnt [4];
    int busy_cycles = 0;
    int late_results = 0;

    always @(posedge clk) begin
        if (rst_n) begin
            for (int j = 0; j < N; j++) begin
                if (sum_valid[j] === 1'b1) begin
                    got[j][got_cnt[j]] = sum_out[j];
                    got_cnt[j] = got_cnt[j] + 1;
                    if (!busy) late_results = late_results + 1;
                end
            end
            if (busy) busy_cycles = busy_cycles + 1;
        end
    end

    task automatic clear_collectors();
        for (int j = 0; j < N; j++) got_cnt[j] = 0;
        busy_cycles  = 0;
        late_results = 0;
    endtask

    //----------------------------------------------------------------
    // Run one tile and check every column against y_exp
    //----------------------------------------------------------------
    task automatic run_and_check(input int tile, input int base,
                                 input int nv, input string name);
        clear_collectors();

        @(negedge clk) begin
            go         = 1;
            w_tile_idx = tile;
            a_base     = base;
            n_vecs     = nv;
        end
        @(negedge clk) go = 0;
        while (busy) @(negedge clk);
        @(negedge clk);   // one settle-out cycle for the collectors

        for (int j = 0; j < N; j++) begin
            if (got_cnt[j] !== nv) begin
                errors++;
                $error("[%s] column %0d: %0d results, expected %0d", name, j, got_cnt[j], nv);
            end else if (got[j][0] !== y_exp[j]) begin
                errors++;
                $error("[%s] y[%0d] = %0d, expected %0d", name, j, got[j][0], y_exp[j]);
            end else begin
                $display("PASS [%s] y[%0d] = %0d", name, j, got[j][0]);
            end
        end
        if (late_results !== 0) begin
            errors++;
            $error("[%s] %0d result(s) arrived after busy fell", name, late_results);
        end
    endtask

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    initial begin
        rst_n = 0; go = 0; w_tile_idx = '0; a_base = '0; n_vecs = '0;
        wr_en_w = 0; wr_addr_w = '0; wr_data_w = '0;
        wr_en_a = 0; wr_addr_a = '0; wr_data_a = '0;

        // golden data
        w1[0][0]=2;  w1[0][1]=-1; w1[0][2]=3;  w1[0][3]=0;
        w1[1][0]=1;  w1[1][1]=4;  w1[1][2]=-2; w1[1][3]=5;
        w1[2][0]=-3; w1[2][1]=2;  w1[2][2]=1;  w1[2][3]=-1;
        w1[3][0]=0;  w1[3][1]=-5; w1[3][2]=2;  w1[3][3]=6;
        a1[0]=7; a1[1]=-3; a1[2]=4; a1[3]=-8;

        w2[0][0]=1;  w2[0][1]=2;  w2[0][2]=3;  w2[0][3]=4;
        w2[1][0]=5;  w2[1][1]=6;  w2[1][2]=7;  w2[1][3]=8;
        w2[2][0]=-1; w2[2][1]=-2; w2[2][2]=-3; w2[2][3]=-4;
        w2[3][0]=9;  w2[3][1]=-9; w2[3][2]=0;  w2[3][3]=1;
        a2[0]=-1; a2[1]=2; a2[2]=-3; a2[3]=4;

        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        //--- T1: tile idx 2, vector at word 7 -------------------------
        for (int i = 0; i < 4; i++)
            write_w(2*K + i, pack4(w1[i][0], w1[i][1], w1[i][2], w1[i][3]));
        write_a(7, pack4(a1[0], a1[1], a1[2], a1[3]));

        // golden: y[j] = sum_i w[i][j]*a[i]
        for (int j = 0; j < 4; j++) begin
            y_exp[j] = 0;
            for (int i = 0; i < 4; i++)
                y_exp[j] = y_exp[j] + w1[i][j]*a1[i];
        end
        run_and_check(2, 7, 1, "T1 matvec tile2");

        // cycle-budget certification (prediction: 17 for 1 vector)
        if (busy_cycles !== PRED_BUSY_1VEC) begin
            errors++;
            $error("[T1 timing] busy = %0d cycles, predicted %0d", busy_cycles, PRED_BUSY_1VEC);
        end else begin
            $display("PASS [T1 timing] busy = %0d cycles (prediction held)", busy_cycles);
        end

        //--- T2: different tile (idx 0), different vector -------------
        for (int i = 0; i < 4; i++)
            write_w(0*K + i, pack4(w2[i][0], w2[i][1], w2[i][2], w2[i][3]));
        write_a(3, pack4(a2[0], a2[1], a2[2], a2[3]));

        for (int j = 0; j < 4; j++) begin
            y_exp[j] = 0;
            for (int i = 0; i < 4; i++)
                y_exp[j] = y_exp[j] + w2[i][j]*a2[i];
        end
        run_and_check(0, 3, 1, "T2 reload tile0");

        //--- T3: back-to-back tiles (step 4, revised) -----------------
        // Buffers already hold tile 2 + vector@7 and tile 0 + vector@3.
        // Issue the second go on the FIRST idle cycle after the first
        // tile; collectors accumulate both tiles' results in order.
        for (int j = 0; j < 4; j++) begin
            y1_exp[j] = 0;
            y2_exp[j] = 0;
            for (int i = 0; i < 4; i++) begin
                y1_exp[j] = y1_exp[j] + w1[i][j]*a1[i];
                y2_exp[j] = y2_exp[j] + w2[i][j]*a2[i];
            end
        end

        clear_collectors();
        @(negedge clk) begin
            go = 1; w_tile_idx = 2; a_base = 7; n_vecs = 1;
        end
        @(negedge clk) go = 0;
        while (busy) @(negedge clk);
        // this IS the single idle cycle: re-issue immediately
        go = 1; w_tile_idx = 0; a_base = 3; n_vecs = 1;
        @(negedge clk) go = 0;
        while (busy) @(negedge clk);
        @(negedge clk);   // settle-out for the collectors

        for (int j = 0; j < 4; j++) begin
            if (got_cnt[j] !== 2) begin
                errors++;
                $error("[T3 b2b] column %0d: %0d results, expected 2", j, got_cnt[j]);
            end else if (got[j][0] !== y1_exp[j] || got[j][1] !== y2_exp[j]) begin
                errors++;
                $error("[T3 b2b] col %0d = (%0d,%0d), expected (%0d,%0d)",
                       j, got[j][0], got[j][1], y1_exp[j], y2_exp[j]);
            end else begin
                $display("PASS [T3 b2b] col %0d = (%0d,%0d)", j, got[j][0], got[j][1]);
            end
        end
        // swap-cost certification: two 17-cycle tiles, 1-cycle gap
        if (busy_cycles !== 2*PRED_BUSY_1VEC) begin
            errors++;
            $error("[T3 timing] busy = %0d cycles, predicted %0d", busy_cycles, 2*PRED_BUSY_1VEC);
        end else begin
            $display("PASS [T3 timing] busy = %0d cycles across both tiles (swap = FLUSH 9 + LOAD 6)", busy_cycles);
        end
        if (late_results !== 0) begin
            errors++;
            $error("[T3 b2b] %0d result(s) arrived after busy fell", late_results);
        end

        //--- summary ---------------------------------------------------
        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
