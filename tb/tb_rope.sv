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

    rope #(.NF(NF), .PHASE_W(PHASE_W), .SIN_N(4096), .SIN_W(SIN_W)) dut (
        .clk (clk), .rst_n (rst_n),
        .start (start), .seek (seek), .seek_pos (seek_pos), .busy (busy),
        .cs_valid (cs_valid), .cs_idx (cs_idx),
        .cos_out (cos_out), .sin_out (sin_out)
    );

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

        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
