`timescale 1ns / 1ps
//====================================================================
// tb_attn_kv_top - the synthesizable attention unit, driven the way a
// decoder actually drives it: one go per token, 102 tokens in a row.
//
//   tb_attn_e2e proved the same modules produce the numpy golden when
//   an initial block sequences them. This proves attn_kv_top's RTL
//   sequencer does, AND that it survives being invoked repeatedly.
//
//   THE REPEATED-INVOCATION PROOF IS THE GOLDEN ITSELF, not a separate
//   assertion. The golden in tb/attn_o*.memh is attention of q against
//   all 102 positions computed in ONE fresh pass. Here the 102nd
//   invocation reaches that same answer after 101 prior invocations
//   have each run a complete scan through the same accumulators. If
//   any per-invocation state leaked - o[], the running max, the
//   running sum, the rescale latch, the not-in-lockstep pos_done
//   latch - the final answer could not land bit-exact. A dedicated
//   "call it twice, check it doesn't crash" test would prove far less.
//
//   C3 then guards the other direction: it shows the INTERMEDIATE
//   invocations were real work and not no-ops, by checking that the
//   1-position result differs from the 102-position golden. Without
//   it, a sequencer that silently scanned all 102 positions every time
//   would also pass C2.
//====================================================================
module tb_attn_kv_top;

    localparam int N   = 128;
    localparam int NQ  = 2;
    localparam int T   = 102;

    logic clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;
    integer errors = 0, i, t, hh, bad;

    logic [7:0]  xk [T*N], xv [T*N];
    logic [31:0] yq0 [N], yq1 [N];
    logic [47:0] eo0 [N], eo1 [N], el0 [1], el1 [1];

    initial begin
        $readmemh("tb/attn_xk.memh",  xk);
        $readmemh("tb/attn_xv.memh",  xv);
        $readmemh("tb/attn_yq0.memh", yq0);
        $readmemh("tb/attn_yq1.memh", yq1);
        $readmemh("tb/attn_o0.memh",  eo0);
        $readmemh("tb/attn_o1.memh",  eo1);
        $readmemh("tb/attn_l0.memh",  el0);
        $readmemh("tb/attn_l1.memh",  el1);
    end

    logic               clear;
    logic               xk_wr, xv_wr, q_wr, go, busy, done;
    logic [6:0]         xk_addr, xv_addr, q_addr, o_rd_addr;
    logic signed [7:0]  xk_data, xv_data;
    logic               q_head, o_rd_head;
    logic signed [31:0] q_data;
    logic [6:0]         n_pos;
    logic               full, overflow;
    logic signed [47:0] o_rd_data;
    logic [47:0]        l_out;
    logic signed [15:0] m_out;

    attn_kv_top #(.N(N), .NQ(NQ), .MAX_POS(T), .SSHIFT(0)) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .xk_wr(xk_wr), .xk_addr(xk_addr), .xk_data(xk_data),
        .xv_wr(xv_wr), .xv_addr(xv_addr), .xv_data(xv_data),
        .q_wr(q_wr), .q_head(q_head), .q_addr(q_addr), .q_data(q_data),
        .go(go), .busy(busy), .done(done),
        .n_pos(n_pos), .full(full), .overflow(overflow),
        .o_rd_head(o_rd_head), .o_rd_addr(o_rd_addr),
        .o_rd_data(o_rd_data), .l_out(l_out), .m_out(m_out));

    // captured after invocation 0, i.e. a 1-position context
    logic [47:0]        l_one;
    logic signed [47:0] o_one;

    task load_token(input integer tt);
        begin
            for (i = 0; i < N; i = i + 1) begin
                @(negedge clk);
                xk_wr = 1'b1; xk_addr = i[6:0]; xk_data = xk[tt*N + i];
                xv_wr = 1'b1; xv_addr = i[6:0]; xv_data = xv[tt*N + i];
            end
            @(negedge clk);
            xk_wr = 1'b0;
            xv_wr = 1'b0;
        end
    endtask

    // A real decoder recomputes q every token, so this reloads every
    // token rather than relying on the registers persisting. Same
    // values each time, so the golden still applies.
    task load_q;
        begin
            for (hh = 0; hh < NQ; hh = hh + 1)
                for (i = 0; i < N; i = i + 1) begin
                    @(negedge clk);
                    q_wr   = 1'b1;
                    q_head = hh[0];
                    q_addr = i[6:0];
                    q_data = (hh == 0) ? yq0[i] : yq1[i];
                end
            @(negedge clk);
            q_wr = 1'b0;
        end
    endtask

    task pulse_go;
        begin
            @(negedge clk); go = 1'b1;
            @(negedge clk); go = 1'b0;
            @(posedge done);
            @(negedge clk);
        end
    endtask

    initial begin
        clear = 0; xk_wr = 0; xv_wr = 0; q_wr = 0; go = 0;
        xk_addr = 0; xv_addr = 0; xk_data = 0; xv_data = 0;
        q_head = 0; q_addr = 0; q_data = 0;
        o_rd_head = 0; o_rd_addr = 0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);
        $display("=== attn_kv_top: %0d sequential go/done invocations ===", T);

        for (t = 0; t < T; t = t + 1) begin
            load_token(t);
            load_q;
            pulse_go;

            // ---- C1: the append happened, exactly once, per go ----
            if (n_pos !== (t+1)) begin
                errors = errors + 1;
                if (errors <= 4)
                    $display("Error: C1 after go %0d n_pos %0d expected %0d",
                             t, n_pos, t+1);
            end
            if (overflow) begin
                errors = errors + 1;
                $display("Error: C1 overflow at go %0d", t);
            end

            if (t == 0) begin
                @(negedge clk); o_rd_head = 1'b0; o_rd_addr = 7'd0;
                @(negedge clk);
                l_one = l_out;
                o_one = o_rd_data;
            end
        end
        $display("  C1 %0d invocations, n_pos tracked every one, full=%b",
                 T, full);

        // ---- C2: the last invocation reproduces the numpy golden ----
        for (hh = 0; hh < NQ; hh = hh + 1) begin
            bad = 0;
            for (i = 0; i < N; i = i + 1) begin
                @(negedge clk);
                o_rd_head = hh[0];
                o_rd_addr = i[6:0];
                @(negedge clk);
                if (o_rd_data !== $signed((hh == 0) ? eo0[i] : eo1[i])) begin
                    bad = bad + 1;
                    if (bad <= 3)
                        $display("Error: C2 head %0d o[%0d] got %012h exp %012h",
                                 hh, i, o_rd_data, (hh == 0) ? eo0[i] : eo1[i]);
                end
            end
            errors = errors + bad;
            @(negedge clk); o_rd_head = hh[0]; @(negedge clk);
            if (l_out !== ((hh == 0) ? el0[0] : el1[0])) begin
                errors = errors + 1;
                $display("Error: C2 head %0d l got %012h exp %012h",
                         hh, l_out, (hh == 0) ? el0[0] : el1[0]);
            end
            $display("  C2 head %0d: %0d/%0d coords bit-exact vs numpy, l %0d",
                     hh, N - bad, N, l_out);
        end

        // ---- C3: the intermediate invocations were not no-ops ----
        if (l_one === el0[0]) begin
            errors = errors + 1;
            $display("Error: C3 1-position l equals the %0d-position golden", T);
        end
        if (o_one === $signed(eo0[0])) begin
            errors = errors + 1;
            $display("Error: C3 1-position o[0] equals the %0d-position golden", T);
        end
        if (l_one === 48'd0) begin
            errors = errors + 1;
            $display("Error: C3 1-position l is zero - invocation 0 did nothing");
        end
        $display("  C3 1-position result l %0d o[0] %0d differs from the %0d-position golden",
                 l_one, o_one, T);

        // ---- C4: clear empties the cache and the unit restarts ----
        @(negedge clk); clear = 1'b1;
        @(negedge clk); clear = 1'b0;
        @(negedge clk);
        if (n_pos !== 7'd0 || full || overflow) begin
            errors = errors + 1;
            $display("Error: C4 after clear n_pos %0d full %b overflow %b",
                     n_pos, full, overflow);
        end
        load_token(0);
        load_q;
        pulse_go;
        if (n_pos !== 7'd1) begin
            errors = errors + 1;
            $display("Error: C4 after clear+go n_pos %0d expected 1", n_pos);
        end
        @(negedge clk); o_rd_head = 1'b0; o_rd_addr = 7'd0;
        @(negedge clk);
        // Position 0 is the same vector as invocation 0 of the sweep,
        // so a correct clear must reproduce that result exactly - a
        // weaker "n_pos went to zero" check would pass even if the
        // stored lines were stale.
        if (l_out !== l_one || o_rd_data !== o_one) begin
            errors = errors + 1;
            $display("Error: C4 post-clear result l %0d/%0d o %0d/%0d",
                     l_out, l_one, o_rd_data, o_one);
        end else begin
            $display("  C4 clear emptied the cache and the rebuilt 1-position result matches");
        end

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

    initial begin
        #2000000000;
        $display("Error: timeout");
        $display("FAILED");
        $finish;
    end

endmodule
