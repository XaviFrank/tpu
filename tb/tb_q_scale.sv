`timescale 1ns / 1ps
//====================================================================
// tb_q_scale - the missing stage between RoPE and attention.
//
//   The golden is tb/attn_yq*.memh: the SAME pre-scaled query the
//   whole verified attention chain already runs on - tb_attn_scan,
//   tb_attn_scan_gqa, tb_attn_e2e, tb_attn_kv_top all read it. So
//   "exact" here means this hardware can replace the numpy line that
//   produced it, which is the only thing that matters.
//
//   Q1  head 6: all 128 coordinates bit-exact against attn_yq0.
//   Q2  head 7 immediately after, with a DIFFERENT (g_mult, g_shift),
//       because the real use is 16 heads in a row each with its own
//       gain and a stale constant would still look plausible.
//   Q3  done coincides with the final y_valid and lands with the FSM
//       idle - the same property rope's rot_done had to be added for.
//   Q4  y_idx: results are captured BY THE INDEX THE DUT REPORTS, and
//       all 128 must appear. An off-by-one in the index pipeline
//       cannot hide behind arrival order.
//   Q5  headroom, reported rather than asserted. The first version of
//       this file tried to test a saturation path; it could not,
//       because at GW=24/QW=32 the rail is unreachable by construction
//       (max |y| = 2**7 * (2**GW - 1) = 2147483520 < 2**31). The clamp
//       was removed and replaced by an elaboration guard in q_scale -
//       so what is left to check here is that real data sits well
//       inside the bound, and the guard itself is checked by building
//       with GW=28, which must fail at time 0.
//====================================================================
module tb_q_scale;

    localparam int N  = 128;
    localparam int GW = 24;
    localparam int SW = 6;
    localparam int QW = 32;

    logic clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    integer errors = 0, i, h, bad;

    logic [7:0]  q8_0 [N], q8_1 [N];
    logic [31:0] gg0  [2], gg1  [2];
    logic [31:0] eyq0 [N], eyq1 [N];

    initial begin
        $readmemh("tb/attn_q8_0.memh", q8_0);
        $readmemh("tb/attn_q8_1.memh", q8_1);
        $readmemh("tb/attn_qg0.memh",  gg0);
        $readmemh("tb/attn_qg1.memh",  gg1);
        $readmemh("tb/attn_yq0.memh",  eyq0);
        $readmemh("tb/attn_yq1.memh",  eyq1);
    end

    logic               go, busy, done;
    logic [GW-1:0]      g_mult;
    logic [SW-1:0]      g_shift;
    logic               x_rd_en;
    logic [6:0]         x_rd_addr;
    logic signed [7:0]  x_rd_data;
    logic               y_valid;
    logic [6:0]         y_idx;
    logic signed [31:0] y_out;

    // query buffer with the same 1-cycle read latency the real source
    // (rope's rotate output, landed in a buffer) will have
    logic head_sel;
    always_ff @(posedge clk) begin
        if (x_rd_en)
            x_rd_data <= head_sel ? $signed(q8_1[x_rd_addr])
                                  : $signed(q8_0[x_rd_addr]);
    end

    q_scale #(.N(N), .GW(GW), .SW(SW), .QW(QW)) dut (
        .clk(clk), .rst_n(rst_n),
        .go(go), .g_mult(g_mult), .g_shift(g_shift),
        .busy(busy), .done(done),
        .x_rd_en(x_rd_en), .x_rd_addr(x_rd_addr), .x_rd_data(x_rd_data),
        .y_valid(y_valid), .y_idx(y_idx), .y_out(y_out));

    logic signed [31:0] got [N];
    logic [N-1:0]       seen;

    always_ff @(posedge clk) begin
        if (y_valid) begin
            got[y_idx]  <= y_out;
            seen[y_idx] <= 1'b1;
        end
    end

    // Q3 monitors, free-running so every invocation is covered
    int yv_at_done, yv_after_done, done_while_busy, done_count, go_count;

    always @(posedge clk) if (rst_n) begin
        if (go && !busy) go_count <= go_count + 1;
        if (done) begin
            done_count <= done_count + 1;
            if (y_valid) yv_at_done     <= yv_at_done + 1;
            if (busy)    done_while_busy <= done_while_busy + 1;
        end
        if (y_valid && !busy && !done) yv_after_done <= yv_after_done + 1;
    end

    task automatic run(input integer hh, input [GW-1:0] gm,
                       input [SW-1:0] gs);
        begin
            @(negedge clk);
            head_sel = hh[0];
            g_mult   = gm;
            g_shift  = gs;
            seen     = '0;
            @(negedge clk); go = 1'b1;
            @(negedge clk); go = 1'b0;
            @(posedge done);
            repeat (2) @(negedge clk);
        end
    endtask

    initial begin
        go = 0; g_mult = 0; g_shift = 0; head_sel = 0; seen = '0;
        yv_at_done = 0; yv_after_done = 0; done_while_busy = 0;
        done_count = 0; go_count = 0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        $display("=== q_scale: INT8 query -> pre-scaled 32-bit ===");

        // ---- Q1/Q2: both heads, back to back, own gains ------------
        for (h = 0; h < 2; h = h + 1) begin
            run(h, (h == 0) ? gg0[0][GW-1:0] : gg1[0][GW-1:0],
                   (h == 0) ? gg0[1][SW-1:0] : gg1[1][SW-1:0]);

            if (seen !== {N{1'b1}}) begin
                errors = errors + 1;
                $display("Error: Q4 head %0d did not emit all %0d indices", h, N);
            end
            bad = 0;
            for (i = 0; i < N; i = i + 1) begin
                if (got[i] !== $signed((h == 0) ? eyq0[i] : eyq1[i])) begin
                    bad = bad + 1;
                    if (bad <= 3)
                        $display("Error: Q%0d head %0d yq[%0d] got %0d exp %0d",
                                 h+1, h, i, got[i],
                                 $signed((h == 0) ? eyq0[i] : eyq1[i]));
                end
            end
            errors = errors + bad;
            $display("  Q%0d head %0d: %0d/%0d bit-exact vs attn_yq%0d, g_mult %0d g_shift %0d",
                     h+1, h, N - bad, N,
                     h, (h == 0) ? gg0[0] : gg1[0], (h == 0) ? gg0[1] : gg1[1]);
        end

        // ---- Q3: done timing ---------------------------------------
        if (done_count !== go_count) begin
            errors = errors + 1;
            $display("Error: Q3 %0d done for %0d go", done_count, go_count);
        end
        if (yv_after_done !== 0) begin
            errors = errors + 1;
            $display("Error: Q3 %0d y_valid beat(s) arrived after done",
                     yv_after_done);
        end
        if (yv_at_done !== done_count) begin
            errors = errors + 1;
            $display("Error: Q3 done coincided with the last beat %0d/%0d times",
                     yv_at_done, done_count);
        end
        if (done_while_busy !== 0) begin
            errors = errors + 1;
            $display("Error: Q3 done asserted %0d time(s) while still busy",
                     done_while_busy);
        end
        $display("  Q3 %0d done pulses, each on the final y_valid, none while busy",
                 done_count);

        // ---- Q5: headroom against the (unreachable) rail -----------
        begin : hdr
            integer mx, v;
            mx = 0;
            for (i = 0; i < N; i = i + 1) begin
                v = got[i];
                if (v < 0) v = -v;
                if (v > mx) mx = v;
            end
            // 2147483520 is max |y| the widths allow; 2147483648 is the
            // rail. Real data should not be anywhere near either.
            if (mx >= 2147483520) begin
                errors = errors + 1;
                $display("Error: Q5 |yq| %0d reached the width bound", mx);
            end
            $display("  Q5 peak |yq| on real data %0d, width bound 2147483520 (rail unreachable by construction)",
                     mx);
        end

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

    initial begin
        #5000000;
        $display("Error: timeout");
        $display("FAILED");
        $finish;
    end

endmodule
