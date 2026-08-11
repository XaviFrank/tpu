`timescale 1ns / 1ps
//====================================================================
// tb_ws_array: golden-model check of input_skew + ws_array driven
// directly (no controller, no BRAM) - the array-level contract.
//   T1: manual weight load, controller protocol: rows fed
//       bottom-first into the parallel column shift chains, one
//       broadcast w_capture when all rows align. Pure-shift load =
//       N+1 = 5 cycles (the step-1 prediction, no BRAM lead-in).
//   T2: TWO activation vectors streamed back-to-back (the
//       forward-compatible batching case): every column delivers
//       both results, in order, on consecutive sum_valid cycles,
//       against y_v[j] = sum_i w[i][j]*a_v[i] (int golden, manual
//       two's complement).
//   T3: output-skew certification: column j's first result arrives
//       exactly j cycles after column 0's.
//====================================================================
module tb_ws_array;

    localparam int N          = 4;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;

    // DUT I/O
    logic clk = 1'b0;
    logic rst_n, load_mode, w_capture, valid_feed;
    logic signed [DATA_WIDTH-1:0] a_feed [N];
    logic signed [ACC_WIDTH-1:0]  b_top  [N];
    logic                         v_bus  [N];
    logic                         v_skewed [N];
    logic signed [DATA_WIDTH-1:0] a_skewed [N];
    logic signed [ACC_WIDTH-1:0]  sum_out   [N];
    logic                         sum_valid [N];

    int errors = 0;

    for (genvar i = 0; i < N; i++) begin : gen_vbus
        assign v_bus[i] = valid_feed;
    end

    input_skew #(.N(N), .DATA_WIDTH(DATA_WIDTH)) u_skew (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (v_bus),
        .data_in   (a_feed),
        .valid_out (v_skewed),
        .data_out  (a_skewed)
    );

    ws_array #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .load_mode (load_mode),
        .w_capture (w_capture),
        .valid_in  (v_skewed),
        .a_in      (a_skewed),
        .b_top     (b_top),
        .sum_out   (sum_out),
        .sum_valid (sum_valid)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    //----------------------------------------------------------------
    // Golden data
    //----------------------------------------------------------------
    int w  [4][4];   // resident weights w[i][j] (row i, column j)
    int a0 [4];      // vector 0
    int a1 [4];      // vector 1
    int y0 [4];
    int y1 [4];

    //----------------------------------------------------------------
    // Collectors: per-column results in arrival order, plus the
    // cycle each column's FIRST result arrived (for the skew check)
    //----------------------------------------------------------------
    int cyc = 0;
    int got       [4][4];
    int got_cnt   [4];
    int first_cyc [4];

    //----------------------------------------------------------------
    // LATENCY INSTRUMENTATION
    //
    // Three timestamps, all taken in THIS always block so they share
    // one cycle convention and the differences are meaningful whatever
    // the absolute offset is:
    //
    //   t_skew_in  first valid at the SKEW's input  (what a TB drives)
    //   t_arr_in   first valid at the ARRAY's left edge (v_skewed)
    //   first_cyc[j]  first sum_valid at the bottom of column j
    //
    // The two input points differ because input_skew gives row i a
    // depth of i+1, not i - so even row 0 costs a cycle, and quoting
    // fill latency from the TB's own drive point overstates the
    // array's by exactly 1.
    //----------------------------------------------------------------
    int t_skew_in = -1;
    int t_arr_in  = -1;

    always @(posedge clk) begin
        cyc = cyc + 1;
        if (rst_n) begin
            for (int i = 0; i < N; i++) begin
                if (t_skew_in < 0 && v_bus[i]    === 1'b1) t_skew_in = cyc;
                if (t_arr_in  < 0 && v_skewed[i] === 1'b1) t_arr_in  = cyc;
            end
            for (int j = 0; j < N; j++) begin
                if (sum_valid[j] === 1'b1) begin
                    if (got_cnt[j] == 0) first_cyc[j] = cyc;
                    got[j][got_cnt[j]] = sum_out[j];
                    got_cnt[j] = got_cnt[j] + 1;
                end
            end
        end
    end

    //----------------------------------------------------------------
    // Drive one weight row onto the top edge (all columns parallel)
    //----------------------------------------------------------------
    task automatic feed_wrow(input int r);
        @(negedge clk) begin
            b_top[0] = w[r][0];
            b_top[1] = w[r][1];
            b_top[2] = w[r][2];
            b_top[3] = w[r][3];
        end
    endtask

    //----------------------------------------------------------------
    // Drive one activation vector (lane i -> row i), valid high.
    // Manual two's complement into the int8 lanes - no $signed.
    //----------------------------------------------------------------
    task automatic feed_vec(input int v0, input int v1,
                            input int v2, input int v3);
        int b0, b1, b2, b3;
        begin
            b0 = (v0 < 0) ? v0 + 256 : v0;
            b1 = (v1 < 0) ? v1 + 256 : v1;
            b2 = (v2 < 0) ? v2 + 256 : v2;
            b3 = (v3 < 0) ? v3 + 256 : v3;
            @(negedge clk) begin
                valid_feed = 1;
                a_feed[0]  = b0[7:0];
                a_feed[1]  = b1[7:0];
                a_feed[2]  = b2[7:0];
                a_feed[3]  = b3[7:0];
            end
        end
    endtask

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    initial begin
        rst_n = 0; load_mode = 0; w_capture = 0; valid_feed = 0;
        for (int i = 0; i < N; i++) begin
            a_feed[i] = '0;
            b_top[i]  = '0;
        end
        for (int j = 0; j < N; j++) got_cnt[j] = 0;

        // golden data
        w[0][0]=1;  w[0][1]=-2; w[0][2]=3;  w[0][3]=4;
        w[1][0]=2;  w[1][1]=0;  w[1][2]=-1; w[1][3]=5;
        w[2][0]=-3; w[2][1]=1;  w[2][2]=2;  w[2][3]=-2;
        w[3][0]=4;  w[3][1]=-5; w[3][2]=0;  w[3][3]=1;
        a0[0]=3;  a0[1]=-1; a0[2]=2; a0[3]=5;
        a1[0]=-2; a1[1]=4;  a1[2]=1; a1[3]=-3;

        for (int j = 0; j < 4; j++) begin
            y0[j] = 0;
            y1[j] = 0;
            for (int i = 0; i < 4; i++) begin
                y0[j] = y0[j] + w[i][j]*a0[i];
                y1[j] = y1[j] + w[i][j]*a1[i];
            end
        end

        repeat (3) @(negedge clk);
        rst_n = 1;

        //--- T1: manual weight load, bottom-first, N+1 cycles ----------
        @(negedge clk) load_mode = 1;
        feed_wrow(3);                       // shift cycle 0
        feed_wrow(2);
        feed_wrow(1);
        feed_wrow(0);                       // shift cycle 3
        @(negedge clk) begin                // alignment cycle: capture
            b_top[0] = '0; b_top[1] = '0; b_top[2] = '0; b_top[3] = '0;
            w_capture = 1;
        end
        @(negedge clk) begin
            w_capture = 0;
            load_mode = 0;
        end
        // spot-check two corners of the resident weight grid
        begin
            int gw;
            gw = {24'b0, dut.gen_row[0].gen_col[0].u_pe.w_reg};
            if (gw >= 128) gw = gw - 256;
            if (gw !== w[0][0]) begin
                errors++; $error("[T1 load] w_reg(0,0) = %0d, expected %0d", gw, w[0][0]);
            end else $display("PASS [T1 load] w_reg(0,0) = %0d", gw);
            gw = {24'b0, dut.gen_row[3].gen_col[1].u_pe.w_reg};
            if (gw >= 128) gw = gw - 256;
            if (gw !== w[3][1]) begin
                errors++; $error("[T1 load] w_reg(3,1) = %0d, expected %0d", gw, w[3][1]);
            end else $display("PASS [T1 load] w_reg(3,1) = %0d", gw);
        end

        //--- T2: two vectors back-to-back ------------------------------
        feed_vec(a0[0], a0[1], a0[2], a0[3]);
        feed_vec(a1[0], a1[1], a1[2], a1[3]);
        @(negedge clk) begin
            valid_feed = 0;
            a_feed[0] = '0; a_feed[1] = '0; a_feed[2] = '0; a_feed[3] = '0;
        end

        // last output (col 3, vector 1) needs 2N+1 cycles after the
        // last feed; wait comfortably past that
        repeat (12) @(negedge clk);

        for (int j = 0; j < 4; j++) begin
            if (got_cnt[j] !== 2) begin
                errors++;
                $error("[T2 b2b vecs] col %0d: %0d results, expected 2", j, got_cnt[j]);
            end else if (got[j][0] !== y0[j] || got[j][1] !== y1[j]) begin
                errors++;
                $error("[T2 b2b vecs] col %0d = (%0d,%0d), expected (%0d,%0d)",
                       j, got[j][0], got[j][1], y0[j], y1[j]);
            end else begin
                $display("PASS [T2 b2b vecs] col %0d = (%0d,%0d)", j, got[j][0], got[j][1]);
            end
        end

        //--- T3: output skew: column j lags column 0 by j cycles -------
        for (int j = 1; j < 4; j++) begin
            if (first_cyc[j] !== first_cyc[0] + j) begin
                errors++;
                $error("[T3 skew] col %0d first result at cyc %0d, expected %0d",
                       j, first_cyc[j], first_cyc[0] + j);
            end else begin
                $display("PASS [T3 skew] col %0d lags col 0 by %0d cycle(s)", j, j);
            end
        end

        //--- T4: LATENCY, measured against the RTL-derived formula -----
        //
        // Derived from ws_pe + ws_array + input_skew, taking cycle 0 as
        // the first valid at the ARRAY's left edge:
        //
        //   ws_pe:  valid_pipe <= valid_in          (+1)
        //           sum_valid  <= valid_pipe        (+1)   -> +2 per PE
        //           a_out      <= a_in              (+1 per column hop)
        //   skew:   row i delayed i+1 at the skew, so at the array's
        //           edge row i leads row 0 by i
        //
        // Row N-1's activation reaches column j at (N-1)+j, and that
        // PE's sum_valid lands 2 cycles later, so the bottom of column
        // j goes valid at N+j+1. Hence
        //
        //   fill (first in -> first out, column 0) = N + 1
        //   end-to-end (first in -> column N-1 out) = 2N
        //   and measured from the SKEW's input instead, both +1.
        //
        // 2N+1 is also exactly ws_controller's FLUSH_CYCLES, which is
        // the cross-check that the drain budget was sized from this
        // same pipeline and not guessed.
        begin : lat
            int fill_arr, e2e_arr, fill_skew, e2e_skew, skew_cost;
            fill_arr  = first_cyc[0]   - t_arr_in;
            e2e_arr   = first_cyc[N-1] - t_arr_in;
            fill_skew = first_cyc[0]   - t_skew_in;
            e2e_skew  = first_cyc[N-1] - t_skew_in;
            skew_cost = t_arr_in - t_skew_in;

            $display("\n--- T4 latency at N=%0d ---", N);
            $display("  skew input -> array input : %0d cycle(s)", skew_cost);
            $display("  FILL      array in -> first out (col 0)   : %0d  (derived N+1 = %0d)",
                     fill_arr, N + 1);
            $display("  END-TO-END array in -> last out (col %0d)  : %0d  (derived 2N = %0d)",
                     N - 1, e2e_arr, 2 * N);
            $display("  same two measured from the skew input     : %0d and %0d",
                     fill_skew, e2e_skew);

            if (skew_cost != 1) begin
                errors++;
                $error("[T4] skew cost %0d, expected 1 (row 0 depth is i+1 = 1)",
                       skew_cost);
            end
            if (fill_arr != N + 1) begin
                errors++;
                $error("[T4] fill %0d, derived %0d", fill_arr, N + 1);
            end
            if (e2e_arr != 2 * N) begin
                errors++;
                $error("[T4] end-to-end %0d, derived %0d", e2e_arr, 2 * N);
            end
            if (skew_cost == 1 && fill_arr == N + 1 && e2e_arr == 2 * N)
                $display("  PASS [T4] both match the RTL-derived formula exactly");
        end

        //--- summary ---------------------------------------------------
        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
