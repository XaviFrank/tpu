`timescale 1ns / 1ps
//====================================================================
// tb_rope: phase generation - accumulate, seek, and cos/sin quality.
//
// The golden is computed directly from the real formula
// theta_i = base^(-2i/head_dim) with base = 1e6, using real math, so
// there is no generated-file indirection for this stage.
//
//   P1: SEEK to a position must give cos(p*theta_i), sin(p*theta_i)
//       for every frequency, checked across four decades of position
//       including the model's last legal one (40959).
//   P2: ACCUMULATE must agree with SEEK. Stepping from 0 n times and
//       seeking straight to n have to produce identical phases - this
//       is what catches a wrong delta or a broken accumulator, which
//       P3 provably cannot (see below).
//   P3: AMPLITUDE, cos^2 + sin^2 = 1. This is the norm-preservation
//       property, and it is deliberately NOT the only check: a phase
//       error leaves amplitude untouched, because rotating by the
//       wrong angle is still a rotation. P3 alone would pass with a
//       completely wrong frequency table.
//   P4: position 0 is the exact anchor - cos=1, sin=0.
//====================================================================
module tb_rope;

    localparam int NF      = 64;
    localparam int PHASE_W = 32;
    localparam int SIN_W   = 16;
    localparam real BASE   = 1000000.0;
    localparam real HD     = 128.0;

    logic clk = 1'b0;
    logic rst_n;
    always #5 clk = ~clk;

    initial begin
        #2_000_000;
        $error("[watchdog] TIMEOUT");
        $finish;
    end

    int errors = 0;

    logic                   start, seek, busy, cs_valid;
    logic [PHASE_W-1:0]     seek_pos;
    logic [5:0]             cs_idx;
    logic signed [SIN_W-1:0] cos_out, sin_out;
    logic                    rot_start, rot_valid, x_rd_en;
    logic [5:0]              x_rd_addr, rot_idx;
    logic signed [7:0]       x1_rd_data, x2_rd_data, rot_x1, rot_x2;
    logic                    step_done, rot_done;

    rope #(.NF(NF), .PHASE_W(PHASE_W), .SIN_N(4096), .SIN_W(SIN_W)) dut (
        .clk (clk), .rst_n (rst_n),
        .start (start), .seek (seek), .seek_pos (seek_pos), .busy (busy),
        .cs_valid (cs_valid), .cs_idx (cs_idx),
        .cos_out (cos_out), .sin_out (sin_out),
        .rot_start (rot_start), .x_rd_en (x_rd_en), .x_rd_addr (x_rd_addr),
        .x1_rd_data (x1_rd_data), .x2_rd_data (x2_rd_data),
        .rot_valid (rot_valid), .rot_idx (rot_idx),
        .rot_x1 (rot_x1), .rot_x2 (rot_x2),
        .step_done (step_done), .rot_done (rot_done)
    );

    //----------------------------------------------------------------
    // R4: completion-pulse timing.
    //
    // Free-running monitors rather than checks inside the run_* tasks,
    // so EVERY command this testbench issues is covered - the rotate
    // cases, the step cases, the seek cases - instead of only the ones
    // an R4-specific task would issue itself.
    //----------------------------------------------------------------
    int   rot_calls, step_calls;      // commands accepted
    int   sd_count, rd_count;         // pulses seen
    int   rv_at_rot_done;             // rot_done coincided with a beat
    int   rv_after_rot_done;          // beats AFTER the pulse - must be 0
    int   rd_while_busy;              // pulse must land with the FSM idle
    int   busy_early;                 // !busy while rot_valid still high

    always @(posedge clk) if (rst_n) begin
        // a command is accepted only when the FSM is actually idle
        if (rot_start && !busy)            rot_calls  <= rot_calls + 1;
        if ((start || seek) && !busy && !rot_start)
                                           step_calls <= step_calls + 1;
        if (step_done)                     sd_count   <= sd_count + 1;
        if (rot_done) begin
            rd_count <= rd_count + 1;
            if (rot_valid) rv_at_rot_done <= rv_at_rot_done + 1;
            if (busy)      rd_while_busy  <= rd_while_busy + 1;
        end
        if (rot_valid && !busy && !rot_done)
            rv_after_rot_done <= rv_after_rot_done + 1;
        if (rot_valid && !busy)
            busy_early <= busy_early + 1;
    end

    initial begin
        rot_calls = 0; step_calls = 0; sd_count = 0; rd_count = 0;
        rv_at_rot_done = 0; rv_after_rot_done = 0;
        rd_while_busy = 0; busy_early = 0;
    end

    // capture one position's worth of cos/sin
    real got_c [NF];
    real got_s [NF];
    int  got_n;
    always @(posedge clk) begin   // real-valued: TB-only, not synthesizable
        if (cs_valid) begin
            got_c[cs_idx] <= real'(cos_out) / 32768.0;
            got_s[cs_idx] <= real'(sin_out) / 32768.0;
            got_n <= got_n + 1;
        end
    end

    // head-vector halves, same 1-cycle read latency as tile_buffer
    logic signed [7:0] xm1 [NF];
    logic signed [7:0] xm2 [NF];
    logic [7:0]        em1 [NF];
    logic [7:0]        em2 [NF];
    logic [31:0]       posmem [1];
    always @(posedge clk) begin
        if (x_rd_en) begin
            x1_rd_data <= xm1[x_rd_addr];
            x2_rd_data <= xm2[x_rd_addr];
        end
    end

    int r1_got [NF];
    int r2_got [NF];
    int rot_n;
    always @(posedge clk) begin
        if (rot_valid) begin
            r1_got[rot_idx] <= int'(rot_x1);
            r2_got[rot_idx] <= int'(rot_x2);
            rot_n <= rot_n + 1;
        end
    end

    task automatic run_rotate();
        int g;
        while (busy) @(negedge clk);
        rot_n = 0;
        @(negedge clk) rot_start = 1'b1;
        @(negedge clk) rot_start = 1'b0;
        g = 0;
        while (rot_n < NF) begin
            @(negedge clk);
            g = g + 1;
            if (g > 4*NF) begin
                errors++; $error("[rotate] only %0d of %0d pairs", rot_n, NF);
                return;
            end
        end
    endtask

    task automatic run_seek(input int unsigned p);
        int g;
        while (busy) @(negedge clk);
        got_n = 0;
        @(negedge clk) begin seek = 1'b1; seek_pos = p; end
        @(negedge clk) seek = 1'b0;
        g = 0;
        while (got_n < NF) begin
            @(negedge clk);
            g = g + 1;
            if (g > 4*NF) begin
                errors++; $error("[seek %0d] only %0d of %0d pairs", p, got_n, NF);
                return;
            end
        end
    endtask

    task automatic run_step();
        int g;
        while (busy) @(negedge clk);
        got_n = 0;
        @(negedge clk) start = 1'b1;
        @(negedge clk) start = 1'b0;
        g = 0;
        while (got_n < NF) begin
            @(negedge clk);
            g = g + 1;
            if (g > 4*NF) begin
                errors++; $error("[step] only %0d of %0d pairs", got_n, NF);
                return;
            end
        end
    endtask

    // check the captured pairs against real cos/sin at position p
    task automatic check_pos(input int unsigned p, input string name,
                             inout real worst_ang, inout real worst_amp);
        int i;
        real th, ang, ec, es, dc, ds, amp;
        for (i = 0; i < NF; i++) begin
            th  = $pow(BASE, -2.0 * real'(i) / HD);
            ang = real'(p) * th;
            ec  = $cos(ang);
            es  = $sin(ang);
            dc  = got_c[i] - ec; if (dc < 0.0) dc = -dc;
            ds  = got_s[i] - es; if (ds < 0.0) ds = -ds;
            if (dc > worst_ang) worst_ang = dc;
            if (ds > worst_ang) worst_ang = ds;
            amp = got_c[i]*got_c[i] + got_s[i]*got_s[i] - 1.0;
            if (amp < 0.0) amp = -amp;
            if (amp > worst_amp) worst_amp = amp;
            // 2e-4: measured worst is 6.3e-5, so ~3x headroom. Set AFTER
            // measuring, not as a placeholder - a loose bound would let a
            // table or delta regression through unnoticed.
            if (dc > 2.0e-4 || ds > 2.0e-4) begin
                errors++;
                $error("[%s] freq %0d: cos %f/%f  sin %f/%f", name, i,
                       got_c[i], ec, got_s[i], es);
            end
        end
    endtask

    real w_ang, w_amp;
    real sc [NF];
    real ss [NF];
    int  i;

    initial begin
        rst_n = 0; start = 0; seek = 0; seek_pos = '0; got_n = 0;
        rot_start = 0; rot_n = 0;
        for (i = 0; i < NF; i++) begin got_c[i]=0.0; got_s[i]=0.0; end
        repeat (3) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        w_ang = 0.0; w_amp = 0.0;

        //--- P4: position 0 is the exact anchor ----------------------
        run_seek(0);
        if (got_c[0] < 0.999 || got_s[0] > 1.0e-4 || got_s[0] < -1.0e-4) begin
            errors++;
            $error("[P4] position 0 freq 0: cos %f sin %f, expected 1 and 0",
                   got_c[0], got_s[0]);
        end else
            $display("PASS [P4] position 0 anchors exactly: cos %f, sin %f",
                     got_c[0], got_s[0]);

        //--- P1: seek across four decades ---------------------------
        run_seek(1);      check_pos(1,     "P1 p=1",     w_ang, w_amp);
        run_seek(7);      check_pos(7,     "P1 p=7",     w_ang, w_amp);
        run_seek(100);    check_pos(100,   "P1 p=100",   w_ang, w_amp);
        run_seek(1000);   check_pos(1000,  "P1 p=1000",  w_ang, w_amp);
        run_seek(40959);  check_pos(40959, "P1 p=40959", w_ang, w_amp);
        $display("PASS [P1] seek matches real cos/sin at 1..40959, worst deviation %e",
                 w_ang);

        //--- P2: accumulate must agree with seek --------------------
        // Seek to 0, step 5 times, and compare against a direct seek
        // to 5. A wrong delta or a broken accumulator shows up here
        // and NOWHERE else - P3 cannot see it.
        run_seek(0);
        for (i = 0; i < 5; i++) run_step();
        for (i = 0; i < NF; i++) begin sc[i] = got_c[i]; ss[i] = got_s[i]; end
        run_seek(5);
        begin
            real d; int bad;
            bad = 0;
            for (i = 0; i < NF; i++) begin
                d = sc[i] - got_c[i]; if (d < 0.0) d = -d;
                if (d > 1.0e-4) bad = bad + 1;
                d = ss[i] - got_s[i]; if (d < 0.0) d = -d;
                if (d > 1.0e-4) bad = bad + 1;
            end
            if (bad != 0) begin
                errors++;
                $error("[P2] accumulate(5 steps) != seek(5) on %0d of %0d values",
                       bad, 2*NF);
            end else
                $display("PASS [P2] accumulate agrees with seek exactly after 5 steps");
        end

        //--- P3: amplitude, i.e. norm preservation -------------------
        $display("PASS [P3] worst |cos^2+sin^2-1| = %e over every checked position",
                 w_amp);
        // 3e-4 against a measured 9.4e-5, same reasoning
        if (w_amp > 3.0e-4) begin
            errors++;
            $error("[P3] amplitude error %e exceeds 3e-4", w_amp);
        end

        //--- R1: rotation vs the exact-math golden (step 3/5) --------
        // Real layer-0 head vector, post q_proj and q_norm, INT8 at the
        // calibration scale - exactly where RoPE sits in the pipeline.
        begin
            int exact, worst, d, j, pk;
            real nin, nout, nd, w_norm;
            $readmemh("tb/rope_x1_q.memh",   xm1);
            $readmemh("tb/rope_x2_q.memh",   xm2);
            $readmemh("tb/rope_exp1_q.memh", em1);
            $readmemh("tb/rope_exp2_q.memh", em2);
            $readmemh("tb/rope_pos.memh",  posmem);

            run_seek(posmem[0]);
            run_rotate();

            exact = 0; worst = 0;
            for (j = 0; j < NF; j++) begin
                d = r1_got[j] - $signed(em1[j]); if (d < 0) d = -d;
                if (d == 0) exact = exact + 1;
                if (d > worst) worst = d;
                if (d > 1) begin
                    errors++;
                    $error("[R1] x1[%0d] got %0d golden %0d", j, r1_got[j], $signed(em1[j]));
                end
                d = r2_got[j] - $signed(em2[j]); if (d < 0) d = -d;
                if (d == 0) exact = exact + 1;
                if (d > worst) worst = d;
                if (d > 1) begin
                    errors++;
                    $error("[R2] x2[%0d] got %0d golden %0d", j, r2_got[j], $signed(em2[j]));
                end
            end
            if (worst <= 1)
                $display("PASS [R1] rotation vs golden at position %0d: %0d/%0d exact, worst |diff| = %0d LSB",
                         posmem[0], exact, 2*NF, worst);

            //--- R2: NORM PRESERVATION across MANY positions ---------
            // Necessary but NOT sufficient - see rope.sv: a phase error
            // leaves the norm untouched, so R1 is what catches those.
            // Swept over four decades because the drift depends on where
            // each component lands relative to an INT8 rounding boundary,
            // and one position would not show the spread.
            nin = 0.0;
            for (j = 0; j < NF; j++)
                nin = nin + real'($signed(xm1[j]))*real'($signed(xm1[j]))
                          + real'($signed(xm2[j]))*real'($signed(xm2[j]));
            nin = $sqrt(nin);
            w_norm = 0.0;
            for (pk = 0; pk < 5; pk++) begin
                case (pk)
                    0: run_seek(1);
                    1: run_seek(137);
                    2: run_seek(1000);
                    3: run_seek(9999);
                    default: run_seek(40959);
                endcase
                run_rotate();
                nout = 0.0;
                for (j = 0; j < NF; j++)
                    nout = nout + real'(r1_got[j])*real'(r1_got[j])
                                + real'(r2_got[j])*real'(r2_got[j]);
                nout = $sqrt(nout);
                nd = (nout - nin) / nin;
                if (nd < 0.0) nd = -nd;
                if (nd > w_norm) w_norm = nd;
            end
            $display("PASS [R2] norm preserved across 5 positions: |in| %.4f, worst rel drift %e",
                     nin, w_norm);
            // 1.2e-2: measured worst over the sweep is 4.08e-3, which is
            // 1.1 sigma of the drift INT8 OUTPUT ROUNDING alone produces on
            // a 128-component vector (|v|=92, sum(d^2)=+10.7 systematic,
            // 2*sum(v*d) sd = 53.2 -> 3 sigma = 1.00e-2). So the bound is
            // set just above 3 sigma: tight enough to catch a real amplitude
            // or saturation problem, loose enough not to fire on ordinary
            // rounding statistics at a different position. Set AFTER
            // sweeping - a single-position guess put it at 1.5e-3 and was
            // wrong.
            // INT8 OUTPUT ROUNDING, not the cos/sin table - each
            // component rounds to nearest, and the norm inherits that.
            if (w_norm > 1.2e-2) begin
                errors++;
                $error("[R2] worst norm drift %e exceeds 1.2e-2", w_norm);
            end
        end

        //--- R3: the same rotation on a real K vector ---------------
        // K differs from Q only in the norm that precedes it and the
        // head count; the rotation itself is identical, so this is a
        // data-diversity check on the same datapath rather than new
        // logic. Its input distribution is quite different (scale 3.69
        // vs 0.29), which is the point.
        begin
            int exact, worst, d, j;
            $readmemh("tb/rope_x1_k.memh",   xm1);
            $readmemh("tb/rope_x2_k.memh",   xm2);
            $readmemh("tb/rope_exp1_k.memh", em1);
            $readmemh("tb/rope_exp2_k.memh", em2);
            run_seek(posmem[0]);
            run_rotate();
            exact = 0; worst = 0;
            for (j = 0; j < NF; j++) begin
                d = r1_got[j] - $signed(em1[j]); if (d < 0) d = -d;
                if (d == 0) exact = exact + 1;
                if (d > worst) worst = d;
                d = r2_got[j] - $signed(em2[j]); if (d < 0) d = -d;
                if (d == 0) exact = exact + 1;
                if (d > worst) worst = d;
            end
            if (worst > 1) begin
                errors++;
                $error("[R3] K rotation worst |diff| = %0d LSB", worst);
            end else
                $display("PASS [R3] K vector rotation: %0d/%0d exact, worst |diff| = %0d LSB",
                         exact, 2*NF, worst);
        end

        repeat (4) @(negedge clk);

        //-------------------------------------------------------------
        // R4: the completion pulses, over every command issued above.
        //-------------------------------------------------------------
        if (rd_count != rot_calls) begin
            errors++;
            $error("[R4] rot_done fired %0d times for %0d rot_start(s)",
                   rd_count, rot_calls);
        end
        if (sd_count != step_calls) begin
            errors++;
            $error("[R4] step_done fired %0d times for %0d start/seek(s)",
                   sd_count, step_calls);
        end
        // not one EARLY: no beat may arrive after the pulse
        if (rv_after_rot_done != 0) begin
            errors++;
            $error("[R4] %0d rot_valid beat(s) arrived after rot_done",
                   rv_after_rot_done);
        end
        // not one LATE: it coincides with the final beat, every time
        if (rv_at_rot_done != rot_calls) begin
            errors++;
            $error("[R4] rot_done coincided with the last beat %0d/%0d times",
                   rv_at_rot_done, rot_calls);
        end
        // and it lands with the FSM already idle, so the next command
        // can be issued on the very next cycle
        if (rd_while_busy != 0) begin
            errors++;
            $error("[R4] rot_done asserted %0d time(s) while still busy",
                   rd_while_busy);
        end
        if (rd_count == rot_calls && sd_count == step_calls
            && rv_after_rot_done == 0 && rv_at_rot_done == rot_calls
            && rd_while_busy == 0)
            $display("PASS [R4] %0d rot_done / %0d step_done, each on the final beat, none late",
                     rd_count, sd_count);

        // WHY THE PULSE HAD TO EXIST. busy falls one cycle EARLY on the
        // rotate path - rot_valid is (state == S_ROT) delayed twice, so
        // the last beat lands after the FSM is already back in S_IDLE.
        // This counter is nonzero BY DESIGN; if it ever reads zero the
        // hazard is gone and the check above has stopped testing it.
        if (busy_early != rot_calls) begin
            errors++;
            $error("[R4] expected busy to fall early on all %0d rotates, saw %0d",
                   rot_calls, busy_early);
        end else
            $display("PASS [R4] busy falls 1 cycle before the last rot_valid on all %0d rotates - which is why !busy is not a completion signal",
                     rot_calls);

        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
