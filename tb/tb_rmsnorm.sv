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
    logic                    g_rd_en;
    logic [AW-1:0]           g_rd_addr;
    logic signed [15:0]      g_rd_data;
    logic                    relu_en;
    logic signed [15:0]      scale_mult;
    logic [4:0]              shift_amt;
    logic                    out_valid;
    logic signed [7:0]       out_data;
    logic [AW-1:0]           out_idx;

    //----------------------------------------------------------------
    // D-CHECK: the new done pulse. Free-running monitors, so every
    // invocation this testbench makes is covered rather than a
    // dedicated case. The property that matters is that done is never
    // EARLY - busy is unusable for chaining (it does not rise until
    // the cycle after go) and a done landing before the last element
    // would lose data just as silently.
    //----------------------------------------------------------------
    logic dut_done;
    int dk_go, dk_done, dk_at, dk_after, dk_busy;
    always @(posedge clk) if (rst_n) begin
        if (go && !busy) dk_go <= dk_go + 1;
        if (dut_done) begin
            dk_done <= dk_done + 1;
            if (out_valid) dk_at   <= dk_at + 1;
            if (busy)      dk_busy <= dk_busy + 1;
        end
        if (out_valid && !busy && !dut_done) dk_after <= dk_after + 1;
    end
    initial begin dk_go = 0; dk_done = 0; dk_at = 0; dk_after = 0; dk_busy = 0; end

    rmsnorm #(.N(N), .IN_W(IN_W), .ACC_W(ACC_W)) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .go          (go),
        .busy        (busy),
        .done        (dut_done),
        .x_rd_en     (x_rd_en),
        .x_rd_addr   (x_rd_addr),
        .x_rd_data   (x_rd_data),
        .eps_fixed   (eps_fixed),
        .ms_valid    (ms_valid),
        .mean_square (mean_square),
        .nrm_valid   (nrm_valid),
        .nrm_mant    (nrm_mant),
        .nrm_shift   (nrm_shift),
        .g_rd_en     (g_rd_en),
        .g_rd_addr   (g_rd_addr),
        .g_rd_data   (g_rd_data),
        .relu_en     (relu_en),
        .scale_mult  (scale_mult),
        .shift_amt   (shift_amt),
        .out_valid   (out_valid),
        .out_data    (out_data),
        .out_idx     (out_idx)
    );


    //----------------------------------------------------------------
    // QK-NORM: the SAME rmsnorm.sv at N=128 with its own GAMMA_F.
    // No new RTL - QK-norm is RMSNorm over head_dim, and the module is
    // already generic in N. The only thing that had to change is the
    // gamma FORMAT: Q2.14 covers the hidden sites (absmax 1.05/1.49)
    // but q_norm reaches 11.81 and k_norm 96.5, which Q2.14 would
    // silently saturate at 2.0. GAMMA_F was already a parameter.
    //----------------------------------------------------------------
    localparam int N2 = 128;
    localparam int AW2 = 7;
    logic                    go2, busy2, ms_valid2, nrm_valid2, out_valid2;
    logic                    lo2;
    assign lo2 = 1'b0;
    logic                    x2_rd_en, g2_rd_en;
    logic [AW2-1:0]          x2_rd_addr, g2_rd_addr, out_idx2;
    logic signed [IN_W-1:0]  x2_rd_data;
    logic signed [15:0]      g2_rd_data;
    logic [ACC_W-1:0]        mean_square2, eps_fixed2;
    logic [17:0]             nrm_mant2;
    logic [5:0]              nrm_shift2;
    logic signed [15:0]      scale_mult2;
    logic [4:0]              shift_amt2;
    logic signed [7:0]       out_data2;

    rmsnorm #(.N(N2), .IN_W(IN_W), .ACC_W(ACC_W), .GAMMA_F(11)) dut2 (
        .clk (clk), .rst_n (rst_n), .go (go2), .busy (busy2),
        .x_rd_en (x2_rd_en), .x_rd_addr (x2_rd_addr), .x_rd_data (x2_rd_data),
        .eps_fixed (eps_fixed2), .ms_valid (ms_valid2), .mean_square (mean_square2),
        .nrm_valid (nrm_valid2), .nrm_mant (nrm_mant2), .nrm_shift (nrm_shift2),
        .g_rd_en (g2_rd_en), .g_rd_addr (g2_rd_addr), .g_rd_data (g2_rd_data),
        .relu_en (lo2), .scale_mult (scale_mult2), .shift_amt (shift_amt2),
        .out_valid (out_valid2), .out_data (out_data2), .out_idx (out_idx2));

    logic signed [IN_W-1:0] xmem2 [N2];
    logic signed [15:0]     gmem2 [N2];
    logic [7:0]             exp2  [N2];
    logic [47:0]            prm2  [4];
    int got2 [N2];
    int got_n2;
    always_ff @(posedge clk) begin
        if (x2_rd_en) x2_rd_data <= xmem2[x2_rd_addr];
        if (g2_rd_en) g2_rd_data <= gmem2[g2_rd_addr];
        if (out_valid2) begin got2[out_idx2] <= int'(out_data2); got_n2 <= got_n2 + 1; end
    end

    // gamma buffer, same 1-cycle latency as the x buffer
    logic signed [15:0] gmem [N];
    always_ff @(posedge clk) begin
        if (g_rd_en) g_rd_data <= gmem[g_rd_addr];
    end

    // collect the INT8 stream by the index the DUT reports
    int got8 [N];
    int got_n;
    always_ff @(posedge clk) begin
        if (out_valid) begin
            got8[out_idx] <= int'(out_data);
            got_n         <= got_n + 1;
        end
    end


    // external x buffer, 1-cycle read latency (tile_buffer's contract)
    logic signed [IN_W-1:0] xmem [N];
    always_ff @(posedge clk) begin
        if (x_rd_en) x_rd_data <= xmem[x_rd_addr];
    end

    task automatic run_and_check(input [ACC_W-1:0] expect_ms, input string name);
        int guard;
        // the DUT runs pass 2 for another N cycles after ms_valid, so a
        // new go must wait for idle - go is only sampled in S_IDLE and
        // would otherwise be silently dropped
        while (busy) @(negedge clk);
            // zero the collector only once the PREVIOUS run has fully
            // drained - every go runs pass 2, so a leftover stream would
            // otherwise inflate got_n and end this run's wait early
            got_n = 0;
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
        // the DUT runs pass 2 for another N cycles after ms_valid, so a
        // new go must wait for idle - go is only sampled in S_IDLE and
        // would otherwise be silently dropped
        while (busy) @(negedge clk);
            // zero the collector only once the PREVIOUS run has fully
            // drained - every go runs pass 2, so a leftover stream would
            // otherwise inflate got_n and end this run's wait early
            got_n = 0;
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

    logic [7:0]  exp8 [N];
    logic [47:0] prm  [4];
    integer i;
    longint unsigned acc_exp;

    initial begin
        rst_n = 0; go = 0; eps_fixed = '0;
        go2 = 1'b0; eps_fixed2 = '0;
        scale_mult2 = 16'sd1; shift_amt2 = 5'd0; got_n2 = 0;
        relu_en = 1'b0; scale_mult = 16'sd1; shift_amt = 5'd0;
        got_n = 0;
        for (i = 0; i < N; i++) begin gmem[i] = 16'sd16384; got8[i] = 0; end
        for (i = 0; i < N2; i++) begin xmem2[i]='0; gmem2[i]='0; got2[i]=0; end
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

        //--- C: FULL PIPELINE vs the verified numpy golden ----------
        // Real Qwen3-0.6B layer-0 site-1 data: a real token embedding
        // as INT16, the real learned input_layernorm gamma as Q2.14,
        // eps and the requant config all derived from real calibration
        // (tb/gen_rmsnorm_vectors.py). The golden comes from the numpy
        // rmsnorm() already verified in the attention work - not
        // re-derived here, per spec section 4.
        begin
            int exact, worst, d, j;
            $readmemh("tb/rmsnorm_x_hidden.memh",      xmem);
            $readmemh("tb/rmsnorm_gamma_hidden.memh",  gmem);
            $readmemh("tb/rmsnorm_exp_hidden.memh",    exp8);
            $readmemh("tb/rmsnorm_params_hidden.memh", prm);
            eps_fixed  = prm[0];
            scale_mult = prm[1][15:0];
            shift_amt  = prm[2][4:0];
            relu_en    = 1'b0;

            while (busy) @(negedge clk);
            // zero the collector only once the PREVIOUS run has fully
            // drained - every go runs pass 2, so a leftover stream would
            // otherwise inflate got_n and end this run's wait early
            got_n = 0;
            @(negedge clk) go = 1'b1;
            @(negedge clk) go = 1'b0;
            j = 0;
            while (got_n < N) begin
                @(negedge clk);
                j = j + 1;
                if (j > 4*N) begin
                    errors++;
                    $error("[C] only %0d of %0d outputs produced", got_n, N);
                    break;
                end
            end

            exact = 0; worst = 0;
            for (j = 0; j < N; j++) begin
                d = got8[j] - $signed(exp8[j][7:0]);
                if (d < 0) d = -d;
                if (d == 0) exact = exact + 1;
                if (d > worst) worst = d;
                if (d > 1) begin
                    errors++;
                    $error("[C] element %0d: got %0d, golden %0d (diff %0d)",
                           j, got8[j], $signed(exp8[j][7:0]), d);
                end
            end
            if (worst <= 1)
                $display("PASS [C] full pipeline vs numpy golden: %0d/%0d exact, worst |diff| = %0d LSB",
                         exact, N, worst);
        end

        //--- E: QK-NORM on the SAME module at N=128 -----------------
        // Real q_norm gamma and a real pre-norm attention head vector.
        // If this passes, QK-norm needs no new RTL at all.
        begin
            int exact, worst, d, j;
            $readmemh("tb/rmsnorm_x_qknorm.memh",      xmem2);
            $readmemh("tb/rmsnorm_gamma_qknorm.memh",  gmem2);
            $readmemh("tb/rmsnorm_exp_qknorm.memh",    exp2);
            $readmemh("tb/rmsnorm_params_qknorm.memh", prm2);
            eps_fixed2  = prm2[0];
            scale_mult2 = prm2[1][15:0];
            shift_amt2  = prm2[2][4:0];

            while (busy2) @(negedge clk);
            got_n2 = 0;
            @(negedge clk) go2 = 1'b1;
            @(negedge clk) go2 = 1'b0;
            j = 0;
            while (got_n2 < N2) begin
                @(negedge clk);
                j = j + 1;
                if (j > 4*N2) begin
                    errors++;
                    $error("[E] only %0d of %0d outputs from the N=128 instance", got_n2, N2);
                    break;
                end
            end
            exact = 0; worst = 0;
            for (j = 0; j < N2; j++) begin
                d = got2[j] - $signed(exp2[j][7:0]);
                if (d < 0) d = -d;
                if (d == 0) exact = exact + 1;
                if (d > worst) worst = d;
                if (d > 1) begin
                    errors++;
                    $error("[E] element %0d: got %0d, golden %0d", j, got2[j], $signed(exp2[j][7:0]));
                end
            end
            if (worst <= 1)
                $display("PASS [E] QK-norm on the SAME module at N=128: %0d/%0d exact, worst |diff| = %0d LSB",
                         exact, N2, worst);
        end

        repeat (4) @(negedge clk);

        // ---- D-CHECK -----------------------------------------------
        if (dk_done != dk_go) begin
            errors++;
            $error("[D] done fired %0d times for %0d go(s)", dk_done, dk_go);
        end
        if (dk_after != 0) begin
            errors++;
            $error("[D] %0d output beat(s) arrived AFTER done", dk_after);
        end
        if (dk_busy != 0) begin
            errors++;
            $error("[D] done asserted %0d time(s) while still busy", dk_busy);
        end
        if (dk_done == dk_go && dk_after == 0 && dk_busy == 0)
            $display("PASS [D] %0d done pulses, none early, none while busy",
                     dk_done);

        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
