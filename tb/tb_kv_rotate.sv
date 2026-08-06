`timescale 1ns / 1ps
//====================================================================
// tb_kv_rotate - both rotation constructions against REAL layer-0
// K and V vectors.
//
//   Vectors come from tb/gen_kv_vectors.py: K post-k_norm and
//   post-RoPE, V raw from v_proj. Qwen3 norms Q and K over head_dim
//   but not V, and RoPE touches Q and K only, so the two arrive at the
//   cache with genuinely different distributions - real outlier
//   structure, not values chosen to suit a quantizer.
//
//   BIT-EXACT, NOT A TOLERANCE. Both rotations are exact integer maps
//   (see kv_rotate.sv header), so there is no rounding convention and
//   no scale in play. A tolerance here would be hiding something.
//
//   R5 checks Parseval - sum(y^2) == N * sum(x^2) exactly - which is
//   an independent structural check on the butterfly wiring. It is
//   NECESSARY BUT NOT SUFFICIENT, exactly as rope.sv's header says of
//   its own norm check: a permuted-but-complete butterfly would still
//   satisfy it. R1-R4 are what actually pin the values down; R5 is
//   there to fail loudly and specifically if a pair index is wrong.
//
//   R6 measures peak-to-rms before and after. That is the property the
//   rotation exists for, so a rotation that produced correct-looking
//   numbers without flattening anything would be a rotation that was
//   not worth building.
//====================================================================
module tb_kv_rotate;

    localparam int N  = 128;
    localparam int AW = 7;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    integer errors = 0;
    integer j;

    //----------------------------------------------------------------
    // Real vectors and their exact goldens
    //----------------------------------------------------------------
    logic [7:0]  x_k     [N];
    logic [7:0]  x_v     [N];
    logic [23:0] e_had_k [N];
    logic [23:0] e_had_v [N];
    logic [23:0] e_pi_k  [N];
    logic [23:0] e_pi_v  [N];

    initial begin
        $readmemh("tb/kv_x_k.memh",   x_k);
        $readmemh("tb/kv_x_v.memh",   x_v);
        $readmemh("tb/kv_had_k.memh", e_had_k);
        $readmemh("tb/kv_had_v.memh", e_had_v);
        $readmemh("tb/kv_pi_k.memh",  e_pi_k);
        $readmemh("tb/kv_pi_v.memh",  e_pi_v);
    end

    //----------------------------------------------------------------
    // Mode 0 - randomized Hadamard
    //----------------------------------------------------------------
    logic [7:0]         buf0 [N];
    logic [23:0]        y0   [N];
    logic               s0_start, s0_busy, s0_rd_en, s0_yv, s0_done;
    logic [AW-1:0]      s0_addr, s0_yidx;
    logic signed [7:0]  s0_rdata;
    logic signed [23:0] s0_yout;

    always_ff @(posedge clk) s0_rdata <= buf0[s0_addr];
    always_ff @(posedge clk) if (s0_yv) y0[s0_yidx] <= s0_yout;

    kv_rotate #(.N(N), .ROT_MODE(0)) dut0 (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (s0_start),
        .busy      (s0_busy),
        .x_rd_en   (s0_rd_en),
        .x_rd_addr (s0_addr),
        .x_rd_data (s0_rdata),
        .y_valid   (s0_yv),
        .y_idx     (s0_yidx),
        .y_out     (s0_yout),
        .done      (s0_done)
    );

    //----------------------------------------------------------------
    // Mode 1 - dense Pi
    //----------------------------------------------------------------
    logic [7:0]         buf1 [N];
    logic [23:0]        y1   [N];
    logic               s1_start, s1_busy, s1_rd_en, s1_yv, s1_done;
    logic [AW-1:0]      s1_addr, s1_yidx;
    logic signed [7:0]  s1_rdata;
    logic signed [23:0] s1_yout;

    always_ff @(posedge clk) s1_rdata <= buf1[s1_addr];
    always_ff @(posedge clk) if (s1_yv) y1[s1_yidx] <= s1_yout;

    kv_rotate #(.N(N), .ROT_MODE(1)) dut1 (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (s1_start),
        .busy      (s1_busy),
        .x_rd_en   (s1_rd_en),
        .x_rd_addr (s1_addr),
        .x_rd_data (s1_rdata),
        .y_valid   (s1_yv),
        .y_idx     (s1_yidx),
        .y_out     (s1_yout),
        .done      (s1_done)
    );

    //----------------------------------------------------------------
    // Manual two's complement. Unpacked-array element signedness is
    // unreliable here, so nothing in this file relies on it.
    //----------------------------------------------------------------
    function automatic longint sext24(input logic [23:0] v);
        longint t;
        begin
            t = v;
            if (v[23]) t = t - 64'd16777216;
            sext24 = t;
        end
    endfunction

    function automatic longint sext8(input logic [7:0] v);
        longint t;
        begin
            t = v;
            if (v[7]) t = t - 64'd256;
            sext8 = t;
        end
    endfunction

    longint sx, sy, tx, ty, mx, my;
    real    cin, cout;

    //----------------------------------------------------------------
    // which: 0 = K, 1 = V
    //----------------------------------------------------------------
    task automatic run0(input integer which);
        integer i;
        begin
            for (i = 0; i < N; i = i + 1)
                buf0[i] = (which == 0) ? x_k[i] : x_v[i];
            @(negedge clk);
            s0_start = 1'b1;
            @(negedge clk);
            s0_start = 1'b0;
            @(posedge s0_done);
            @(posedge clk);
            @(posedge clk);
        end
    endtask

    task automatic run1(input integer which);
        integer i;
        begin
            for (i = 0; i < N; i = i + 1)
                buf1[i] = (which == 0) ? x_k[i] : x_v[i];
            @(negedge clk);
            s1_start = 1'b1;
            @(negedge clk);
            s1_start = 1'b0;
            @(posedge s1_done);
            @(posedge clk);
            @(posedge clk);
        end
    endtask

    task automatic check0(input integer which, input [127:0] tag);
        integer i;
        logic [23:0] exp;
        integer bad;
        begin
            bad = 0;
            for (i = 0; i < N; i = i + 1) begin
                exp = (which == 0) ? e_had_k[i] : e_had_v[i];
                if (y0[i] !== exp) begin
                    bad = bad + 1;
                    if (bad <= 5)
                        $display("Error: %0s idx %0d got %h exp %h",
                                 tag, i, y0[i], exp);
                end
            end
            errors = errors + bad;
            $display("  %0s  %0d/%0d exact", tag, N - bad, N);
        end
    endtask

    task automatic check1(input integer which, input [127:0] tag);
        integer i;
        logic [23:0] exp;
        integer bad;
        begin
            bad = 0;
            for (i = 0; i < N; i = i + 1) begin
                exp = (which == 0) ? e_pi_k[i] : e_pi_v[i];
                if (y1[i] !== exp) begin
                    bad = bad + 1;
                    if (bad <= 5)
                        $display("Error: %0s idx %0d got %h exp %h",
                                 tag, i, y1[i], exp);
                end
            end
            errors = errors + bad;
            $display("  %0s  %0d/%0d exact", tag, N - bad, N);
        end
    endtask

    //----------------------------------------------------------------
    // Parseval + crest, on whichever mode-0 run just completed.
    //----------------------------------------------------------------
    task automatic stats0(input [127:0] tag);
        integer i;
        begin
            sx = 0; sy = 0; mx = 0; my = 0;
            for (i = 0; i < N; i = i + 1) begin
                tx = sext8(buf0[i]);
                ty = sext24(y0[i]);
                sx = sx + tx * tx;
                sy = sy + ty * ty;
                if (tx < 0) tx = -tx;
                if (ty < 0) ty = -ty;
                if (tx > mx) mx = tx;
                if (ty > my) my = ty;
            end
            if (sy !== N * sx) begin
                errors = errors + 1;
                $display("Error: %0s Parseval sum(y^2)=%0d != %0d*sum(x^2)=%0d",
                         tag, sy, N, N * sx);
            end else begin
                $display("  %0s  Parseval exact: sum(y^2) = %0d = %0d * %0d",
                         tag, sy, N, sx);
            end
            cin  = mx / $sqrt(sx * 1.0 / N);
            cout = my / $sqrt(sy * 1.0 / N);
            $display("  %0s  peak/rms  in %0.3f -> out %0.3f", tag, cin, cout);
            if (cout >= cin) begin
                errors = errors + 1;
                $display("Error: %0s rotation did not flatten (%0.3f -> %0.3f)",
                         tag, cin, cout);
            end
        end
    endtask

    task automatic stats1(input [127:0] tag);
        integer i;
        begin
            sx = 0; sy = 0; mx = 0; my = 0;
            for (i = 0; i < N; i = i + 1) begin
                tx = sext8(buf1[i]);
                ty = sext24(y1[i]);
                sx = sx + tx * tx;
                sy = sy + ty * ty;
                if (tx < 0) tx = -tx;
                if (ty < 0) ty = -ty;
                if (tx > mx) mx = tx;
                if (ty > my) my = ty;
            end
            cin  = mx / $sqrt(sx * 1.0 / N);
            cout = my / $sqrt(sy * 1.0 / N);
            $display("  %0s  peak/rms  in %0.3f -> out %0.3f", tag, cin, cout);
            if (cout >= cin) begin
                errors = errors + 1;
                $display("Error: %0s rotation did not flatten (%0.3f -> %0.3f)",
                         tag, cin, cout);
            end
        end
    endtask

    //----------------------------------------------------------------
    initial begin
        s0_start = 1'b0;
        s1_start = 1'b0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        $display("=== kv_rotate: real layer-0 K and V ===");

        $display("R1/R2 randomized Hadamard (ROT_MODE=0), exact integer");
        run0(0);  check0(0, "R1 K had"); stats0("R1 K had");
        run0(1);  check0(1, "R2 V had"); stats0("R2 V had");

        $display("R3/R4 dense Pi (ROT_MODE=1), exact integer");
        run1(0);  check1(0, "R3 K pi ");  stats1("R3 K pi ");
        run1(1);  check1(1, "R4 V pi ");  stats1("R4 V pi ");

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

    initial begin
        #4000000;
        $display("Error: timeout");
        $display("FAILED");
        $finish;
    end

endmodule
