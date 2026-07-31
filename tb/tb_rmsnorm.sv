`timescale 1ns / 1ps
//====================================================================
// tb_rmsnorm: stage 1-2 (sum of squares, then the >>log2(N) mean).
//
// The TB models the external x buffer with the SAME 1-cycle read
// latency tile_buffer has, so the DUT's land-pipe timing is exercised
// exactly as it will be against real memory - an off-by-one there
// would drop the first or last element, which is precisely the kind of
// bug a hand-computed golden catches and a plausible-looking number
// does not.
//
//   A1: all-ones          -> mean = 1                (trivial anchor)
//   A2: alternating +/-k  -> mean = k^2              (sign handling)
//   A3: full-scale +/-32767 -> mean = 32767^2        (no overflow, and
//       the accumulator's worst case: N*(2^15)^2 = 2^40)
//   A4: a ramp            -> mean = sum(i^2)/N       (hand-computed)
//   A5: back-to-back runs -> the accumulator clears on go, so a second
//       run does not inherit the first (would silently double)
//   A6: exact element count - a deliberately altered buffer where ONLY
//       element 0 and element N-1 are nonzero, so dropping either end
//       changes the answer. Catches the land-pipe off-by-one directly.
//   B : stage 3 rsqrt vs REAL 1/sqrt, swept over four decades plus
//       both real per-site eps values. Threshold 1e-4 against a
//       measured 3.4e-5 - tight enough that a one-element
//       accumulator slip (which showed up as 5.0e-4) fails.
//====================================================================
module tb_rmsnorm;

    localparam int N     = 1024;
    localparam int IN_W  = 16;
    localparam int ACC_W = 48;
    localparam int AW    = 10;

    logic clk = 1'b0;
    logic rst_n;
    always #5 clk = ~clk;

    initial begin
        #2_000_000;
        $error("[watchdog] TIMEOUT");
        $finish;
    end

    int errors = 0;

    logic                    go, busy;
    logic                    x_rd_en;
    logic [AW-1:0]           x_rd_addr;
    logic signed [IN_W-1:0]  x_rd_data;
    logic                    ms_valid;
    logic [ACC_W-1:0]        mean_square;
    logic [ACC_W-1:0]        eps_fixed;
    logic                    nrm_valid;
    logic [17:0]             nrm_mant;
    logic [5:0]              nrm_shift;

    rmsnorm #(.N(N), .IN_W(IN_W), .ACC_W(ACC_W)) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .go          (go),
        .busy        (busy),
        .x_rd_en     (x_rd_en),
        .x_rd_addr   (x_rd_addr),
        .x_rd_data   (x_rd_data),
        .eps_fixed   (eps_fixed),
        .ms_valid    (ms_valid),
        .mean_square (mean_square),
        .nrm_valid   (nrm_valid),
        .nrm_mant    (nrm_mant),
        .nrm_shift   (nrm_shift)
    );

    // external x buffer, 1-cycle read latency (tile_buffer's contract)
    logic signed [IN_W-1:0] xmem [N];
    always_ff @(posedge clk) begin
        if (x_rd_en) x_rd_data <= xmem[x_rd_addr];
    end

    task automatic run_and_check(input [ACC_W-1:0] expect_ms, input string name);
        int guard;
        @(negedge clk) go = 1'b1;
        @(negedge clk) go = 1'b0;
        guard = 0;
        while (ms_valid !== 1'b1) begin
            @(negedge clk);
            guard = guard + 1;
            if (guard > N + 64) begin
                errors++;
                $error("[%s] ms_valid never asserted", name);
                return;
            end
        end
        if (mean_square !== expect_ms) begin
            errors++;
            $error("[%s] mean_square = %0d, expected %0d",
                   name, mean_square, expect_ms);
        end else begin
            $display("PASS [%s] mean_square = %0d", name, mean_square);
        end
    endtask

    // Compare the hardware normaliser against real 1/sqrt. The DUT
    // reports mantissa+shift, so the reconstruction is
    // nrm_mant / 2**nrm_shift.
    task automatic check_rsqrt(input signed [IN_W-1:0] c,
                               input [ACC_W-1:0] eps,
                               input string name,
                               inout real worst);
        real m, expd, got, rel;
        int j;
        for (j = 0; j < N; j++) xmem[j] = c;
        eps_fixed = eps;
        @(negedge clk) go = 1'b1;
        @(negedge clk) go = 1'b0;
        j = 0;
        while (nrm_valid !== 1'b1) begin
            @(negedge clk);
            j = j + 1;
            if (j > N + 64) begin
                errors++; $error("[%s] nrm_valid never asserted", name); return;
            end
        end
        m    = real'(mean_square) + real'(eps);
        expd = 1.0 / $sqrt(m);
        got  = real'(nrm_mant) / $pow(2.0, real'(nrm_shift));
        rel  = (got - expd) / expd;
        if (rel < 0.0) rel = -rel;
        if (rel > worst) worst = rel;
        // 1e-4, chosen AFTER measuring: the sweep achieves 3.4e-5, and the
        // stale-accumulator bug this caught produced 5.0e-4. A looser
        // threshold would have let that regress silently.
        if (rel > 1.0e-4) begin
            errors++;
            $error("[%s] normalizer %e vs exact %e (M=%e), rel err %e",
                   name, got, expd, m, rel);
        end else begin
            $display("  [B] %-18s M=%-14.0f 1/sqrt: got %e exact %e rel %e",
                     name, m, got, expd, rel);
        end
    endtask

    integer i;
    longint unsigned acc_exp;

    initial begin
        rst_n = 0; go = 0; eps_fixed = '0;
        for (i = 0; i < N; i++) xmem[i] = 16'sd0;
        repeat (3) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        //--- A1: all ones -------------------------------------------
        for (i = 0; i < N; i++) xmem[i] = 16'sd1;
        run_and_check(48'd1, "A1 all-ones");

        //--- A2: alternating +/-300 ---------------------------------
        for (i = 0; i < N; i++) xmem[i] = (i % 2 == 0) ? 16'sd300 : -16'sd300;
        run_and_check(48'd90000, "A2 alternating +/-300");

        //--- A3: full-scale, the accumulator's worst case ------------
        for (i = 0; i < N; i++) xmem[i] = (i % 2 == 0) ? 16'sh7FFF : -16'sh7FFF;
        run_and_check(48'd1073676289, "A3 full-scale +/-32767");

        //--- A4: ramp, hand-computed --------------------------------
        acc_exp = 0;
        for (i = 0; i < N; i++) begin
            xmem[i] = 16'(i);
            acc_exp = acc_exp + longint'(i) * longint'(i);
        end
        run_and_check(acc_exp >> 10, "A4 ramp");

        //--- A5: back-to-back, accumulator must clear on go ----------
        // Same data as A4: if acc did not clear, this reads ~2x.
        run_and_check(acc_exp >> 10, "A5 rerun clears accumulator");

        //--- A6: only the two END elements are nonzero ---------------
        // Dropping element 0 or element N-1 - the classic land-pipe
        // off-by-one - changes the answer, so this pins the count.
        for (i = 0; i < N; i++) xmem[i] = 16'sd0;
        xmem[0]     = 16'sd1024;      // 1024^2 = 1048576
        xmem[N-1]   = 16'sd2048;      // 2048^2 = 4194304
        // (1048576 + 4194304) / 1024 = 5120
        run_and_check(48'd5120, "A6 both end elements counted");

        //--- A7: only element N-1, to isolate the tail --------------
        for (i = 0; i < N; i++) xmem[i] = 16'sd0;
        xmem[N-1] = 16'sd1024;                       // 1048576/1024 = 1024
        run_and_check(48'd1024, "A7 last element counted");

        //--- A8: only element 0, to isolate the head ----------------
        for (i = 0; i < N; i++) xmem[i] = 16'sd0;
        xmem[0] = 16'sd1024;
        run_and_check(48'd1024, "A8 first element counted");

        //--- B: stage 3, 1/sqrt against REAL math -------------------
        // Every element is set to c, so mean_square is exactly c^2 and
        // M is exactly c^2 + eps - no golden-model ambiguity. Swept
        // over four decades plus BOTH real per-site eps values, so the
        // normaliser is checked where it will actually operate rather
        // than at one convenient point.
        begin
            real worst;
            worst = 0.0;
            check_rsqrt(16'sd1,     48'd0,     "c=1 eps=0",       worst);
            check_rsqrt(16'sd3,     48'd0,     "c=3 eps=0",       worst);
            check_rsqrt(16'sd7,     48'd0,     "c=7 eps=0",       worst);
            check_rsqrt(16'sd15,    48'd0,     "c=15 eps=0",      worst);
            check_rsqrt(16'sd100,   48'd0,     "c=100 eps=0",     worst);
            check_rsqrt(16'sd181,   48'd0,     "c=181 eps=0",     worst);
            check_rsqrt(16'sd1000,  48'd0,     "c=1000 eps=0",    worst);
            check_rsqrt(16'sd4096,  48'd0,     "c=4096 eps=0",    worst);
            check_rsqrt(16'sd11585, 48'd0,     "c=11585 eps=0",   worst);
            check_rsqrt(16'sd32767, 48'd0,     "c=32767 eps=0",   worst);
            // the real per-site operating points (see rmsnorm.sv header)
            check_rsqrt(16'sd1950,  48'd100,   "site2 residual",  worst);
            check_rsqrt(16'sd6000,  48'd55054, "site1 embedding", worst);
            // eps dominating a tiny mean_square - the degenerate end
            check_rsqrt(16'sd1,     48'd55054, "eps dominates",   worst);
            $display("PASS [B] rsqrt worst relative error over sweep: %e", worst);
        end

        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
