`timescale 1ns / 1ps
//====================================================================
// tb_attn_kv_layers - two layers sharing one cache, interleaved the
// way a 28-layer decode interleaves them.
//
//   tb_attn_kv_top proves the numerics, at one layer, against numpy.
//   This proves the thing that test cannot: that a SECOND layer using
//   the same store does not disturb the first. The failure mode being
//   hunted is aliasing - a layer offset dropped, truncated, or applied
//   on one port and not the other - which produces a perfectly
//   self-consistent wrong answer, so no amount of single-layer testing
//   finds it.
//
//   THE REFERENCE IS THE SAME DUT, NOT A SECOND MODEL. Phases 1 and 2
//   run each layer's token stream ALONE and capture the result; phase
//   3 replays both interleaved and demands the identical answer. That
//   is the definition of isolation stated directly, and it needs no
//   golden of its own - the absolute numerics are already nailed down
//   by tb_attn_kv_top's numpy comparison.
//
//   The two streams carry DIFFERENT real vectors (tokens 0..TL-1 and
//   TL..2TL-1). Feeding both layers the same data would pass even if
//   the offset were ignored entirely and both layers aliased onto the
//   same region, which is the single most likely bug here.
//
//   Phase 3 interleaves as L0,L1 per token and then finishes with the
//   final L1 token and the final L0 token in that order, so BOTH
//   layers' complete results can be read out of the single set of
//   scan accumulators before the other overwrites them.
//====================================================================
module tb_attn_kv_layers;

    localparam int N   = 128;
    localparam int NQ  = 2;
    localparam int TL  = 8;     // positions per layer
    localparam int NL  = 2;
    localparam int TFILE = 102;  // rows in tb/attn_x*.memh; only 2*TL used

    logic clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;
    integer errors = 0, i, t, hh, bad;

    logic [7:0]  xk [TFILE*N], xv [TFILE*N];
    logic [31:0] yq0 [N], yq1 [N];

    initial begin
        $readmemh("tb/attn_xk.memh",  xk);
        $readmemh("tb/attn_xv.memh",  xv);
        $readmemh("tb/attn_yq0.memh", yq0);
        $readmemh("tb/attn_yq1.memh", yq1);
    end

    logic               clear, layer;
    logic               xk_wr, xv_wr, q_wr, go, busy, done;
    logic [6:0]         xk_addr, xv_addr, q_addr, o_rd_addr;
    logic signed [7:0]  xk_data, xv_data;
    logic               q_head, o_rd_head;
    logic signed [31:0] q_data;
    logic [3:0]         n_pos;
    logic               full, overflow;
    logic signed [47:0] o_rd_data;
    logic [47:0]        l_out;
    logic signed [15:0] m_out;

    attn_kv_top #(.N(N), .NQ(NQ), .MAX_POS(TL), .NUM_LAYERS(NL),
                  .SSHIFT(0)) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .layer(layer),
        .xk_wr(xk_wr), .xk_addr(xk_addr), .xk_data(xk_data),
        .xv_wr(xv_wr), .xv_addr(xv_addr), .xv_data(xv_data),
        .q_wr(q_wr), .q_head(q_head), .q_addr(q_addr), .q_data(q_data),
        .go(go), .busy(busy), .done(done),
        .n_pos(n_pos), .full(full), .overflow(overflow),
        .o_rd_head(o_rd_head), .o_rd_addr(o_rd_addr),
        .o_rd_data(o_rd_data), .l_out(l_out), .m_out(m_out));

    // reference captures: [layer][head][coord], plus l per (layer,head)
    logic signed [47:0] ref_o [NL*NQ*N];
    logic [47:0]        ref_l [NL*NQ];

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

    // one invocation on layer `ly` with token `tt`
    task invoke(input integer ly, input integer tt);
        begin
            @(negedge clk);
            layer = ly[0];
            load_token(tt);
            load_q;
            @(negedge clk); go = 1'b1;
            @(negedge clk); go = 1'b0;
            @(posedge done);
            @(negedge clk);
        end
    endtask

    task capture(input integer ly);
        begin
            for (hh = 0; hh < NQ; hh = hh + 1) begin
                for (i = 0; i < N; i = i + 1) begin
                    @(negedge clk);
                    o_rd_head = hh[0];
                    o_rd_addr = i[6:0];
                    @(negedge clk);
                    ref_o[(ly*NQ + hh)*N + i] = o_rd_data;
                end
                @(negedge clk); o_rd_head = hh[0]; @(negedge clk);
                ref_l[ly*NQ + hh] = l_out;
            end
        end
    endtask

    task compare(input integer ly, output integer nbad);
        begin
            nbad = 0;
            for (hh = 0; hh < NQ; hh = hh + 1) begin
                for (i = 0; i < N; i = i + 1) begin
                    @(negedge clk);
                    o_rd_head = hh[0];
                    o_rd_addr = i[6:0];
                    @(negedge clk);
                    if (o_rd_data !== ref_o[(ly*NQ + hh)*N + i]) begin
                        nbad = nbad + 1;
                        if (nbad <= 3)
                            $display("Error: L3 layer %0d head %0d o[%0d] got %012h alone %012h",
                                     ly, hh, i, o_rd_data,
                                     ref_o[(ly*NQ + hh)*N + i]);
                    end
                end
                @(negedge clk); o_rd_head = hh[0]; @(negedge clk);
                if (l_out !== ref_l[ly*NQ + hh]) begin
                    nbad = nbad + 1;
                    $display("Error: L3 layer %0d head %0d l got %012h alone %012h",
                             ly, hh, l_out, ref_l[ly*NQ + hh]);
                end
            end
        end
    endtask

    task do_clear;
        begin
            @(negedge clk); clear = 1'b1;
            @(negedge clk); clear = 1'b0;
            @(negedge clk);
        end
    endtask

    integer d0, d1, ndiff;

    initial begin
        clear = 0; layer = 0; xk_wr = 0; xv_wr = 0; q_wr = 0; go = 0;
        xk_addr = 0; xv_addr = 0; xk_data = 0; xv_data = 0;
        q_head = 0; q_addr = 0; q_data = 0;
        o_rd_head = 0; o_rd_addr = 0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);
        $display("=== attn_kv_layers: %0d layers x %0d positions, one shared cache ===",
                 NL, TL);

        // ---- L1: layer 0's stream, alone ----------------------------
        do_clear;
        for (t = 0; t < TL; t = t + 1) invoke(0, t);
        if (n_pos !== TL[3:0] || overflow) begin
            errors = errors + 1;
            $display("Error: L1 n_pos %0d overflow %b", n_pos, overflow);
        end
        capture(0);
        $display("  L1 layer 0 alone: n_pos %0d, l %0d", n_pos, ref_l[0]);

        // ---- L2: layer 1's stream, alone ----------------------------
        do_clear;
        for (t = 0; t < TL; t = t + 1) invoke(1, TL + t);
        if (n_pos !== TL[3:0] || overflow) begin
            errors = errors + 1;
            $display("Error: L2 n_pos %0d overflow %b", n_pos, overflow);
        end
        capture(1);
        $display("  L2 layer 1 alone: n_pos %0d, l %0d", n_pos, ref_l[NQ]);

        // The two streams must actually differ, or L3 proves nothing.
        ndiff = 0;
        for (i = 0; i < N; i = i + 1)
            if (ref_o[i] !== ref_o[NQ*N + i]) ndiff = ndiff + 1;
        if (ndiff == 0) begin
            errors = errors + 1;
            $display("Error: L2 the two layers' reference results are identical - L3 is vacuous");
        end
        $display("  L2 the two reference results differ in %0d/%0d coords", ndiff, N);

        // ---- L3: both interleaved, in one cache ---------------------
        do_clear;
        for (t = 0; t < TL-1; t = t + 1) begin
            invoke(0, t);
            invoke(1, TL + t);
        end
        invoke(1, TL + TL - 1);
        compare(1, d1);
        invoke(0, TL - 1);
        compare(0, d0);

        errors = errors + d0 + d1;
        if (overflow) begin
            errors = errors + 1;
            $display("Error: L3 overflow asserted during the interleave");
        end
        $display("  L3 interleaved: layer 0 %0d mismatches, layer 1 %0d mismatches",
                 d0, d1);

        // ---- L4: per-layer position counts are independent ----------
        @(negedge clk); layer = 1'b0; @(negedge clk);
        if (n_pos !== TL[3:0] || !full) begin
            errors = errors + 1;
            $display("Error: L4 layer 0 n_pos %0d full %b expected %0d 1",
                     n_pos, full, TL);
        end
        @(negedge clk); layer = 1'b1; @(negedge clk);
        if (n_pos !== TL[3:0] || !full) begin
            errors = errors + 1;
            $display("Error: L4 layer 1 n_pos %0d full %b expected %0d 1",
                     n_pos, full, TL);
        end
        do_clear;
        @(negedge clk); layer = 1'b0; @(negedge clk);
        if (n_pos !== 4'd0 || full) begin
            errors = errors + 1;
            $display("Error: L4 clear left layer 0 at n_pos %0d full %b", n_pos, full);
        end
        @(negedge clk); layer = 1'b1; @(negedge clk);
        if (n_pos !== 4'd0 || full) begin
            errors = errors + 1;
            $display("Error: L4 clear left layer 1 at n_pos %0d full %b", n_pos, full);
        end
        $display("  L4 per-layer n_pos readable while idle, clear empties both");

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

    initial begin
        #200000000;
        $display("Error: timeout");
        $display("FAILED");
        $finish;
    end

endmodule
