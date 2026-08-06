`timescale 1ns / 1ps
//====================================================================
// tb_attn_scan_gqa - one KV-head scan serving both sharing Q heads.
//
//   ONE K stream and ONE V stream are broadcast to both heads, which
//   is the entire point: on a bandwidth-bound read side the cost is
//   fetching, and re-scanning per Q head would double DRAM traffic.
//
//   G1/G2 compare each head against a golden computed INDEPENDENTLY in
//   numpy, with no shared state - which is exactly section 6's
//   requirement that reuse must not corrupt one head while computing
//   the other. A bug that mixed the two accumulators would still
//   produce plausible numbers; only an independent per-head reference
//   catches it.
//
//   The two heads rescale a DIFFERENT number of times on this row
//   (1 and 6), so they do not run in lockstep - which makes this also
//   a test that the shared sequencer waits for the slower head.
//====================================================================
module tb_attn_scan_gqa;

    localparam int N   = 128;
    localparam int NQ  = 2;
    localparam int T   = 102;
    localparam int NWK = 32;
    localparam int NWV = 13;

    logic clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;
    integer errors = 0, i, t, hh, bad;

    logic [31:0] yq0 [N], yq1 [N];
    logic [31:0] kw [T*NWK], vw [T*NWV];
    logic [47:0] eo0 [N], eo1 [N], el0 [1], el1 [1];

    initial begin
        $readmemh("tb/attn_yq0.memh", yq0);
        $readmemh("tb/attn_yq1.memh", yq1);
        $readmemh("tb/attn_kw.memh",  kw);
        $readmemh("tb/attn_vw.memh",  vw);
        $readmemh("tb/attn_o0.memh",  eo0);
        $readmemh("tb/attn_o1.memh",  eo1);
        $readmemh("tb/attn_l0.memh",  el0);
        $readmemh("tb/attn_l1.memh",  el1);
    end

    logic q_wr, prep_start, prep_done, pos_last, busy, pos_done;
    logic q_head, o_rd_head;
    logic [6:0] q_addr, o_rd_addr;
    logic signed [31:0] q_data;
    logic k_wv, v_wv;
    logic [31:0] k_wd, v_wd;
    logic signed [47:0] o_rd_data;
    logic [47:0] l_out;
    logic signed [15:0] m_out;

    attn_scan_gqa #(.N(N), .NQ(NQ), .SSHIFT(0)) dut (
        .clk(clk), .rst_n(rst_n),
        .q_wr(q_wr), .q_head(q_head), .q_addr(q_addr), .q_data(q_data),
        .prep_start(prep_start), .prep_done(prep_done),
        .pos_last(pos_last), .busy(busy), .pos_done(pos_done),
        .k_w_valid(k_wv), .k_w_data(k_wd),
        .v_w_valid(v_wv), .v_w_data(v_wd),
        .o_rd_head(o_rd_head), .o_rd_addr(o_rd_addr),
        .o_rd_data(o_rd_data), .l_out(l_out), .m_out(m_out));

    initial begin
        q_wr=0; prep_start=0; pos_last=0; k_wv=0; v_wv=0;
        q_head=0; o_rd_head=0; q_addr=0; o_rd_addr=0; q_data=0;
        repeat (4) @(negedge clk); rst_n = 1'b1; repeat (2) @(negedge clk);
        $display("=== attn_scan_gqa: 1 KV scan -> %0d Q heads, %0d positions ===", NQ, T);

        for (hh = 0; hh < NQ; hh = hh + 1)
            for (i = 0; i < N; i = i + 1) begin
                @(negedge clk);
                q_wr = 1'b1; q_head = hh[0]; q_addr = i[6:0];
                q_data = (hh == 0) ? yq0[i] : yq1[i];
            end
        @(negedge clk); q_wr = 1'b0;
        @(negedge clk); prep_start = 1'b1;
        @(negedge clk); prep_start = 1'b0;
        @(posedge prep_done); repeat (2) @(negedge clk);

        for (t = 0; t < T; t = t + 1) begin
            pos_last = (t == T-1);
            while (busy) @(negedge clk);
            for (i = 0; i < NWK; i = i + 1) begin
                @(negedge clk); k_wv = 1'b1; k_wd = kw[t*NWK + i];
            end
            @(negedge clk); k_wv = 1'b0;
            for (i = 0; i < NWV; i = i + 1) begin
                @(negedge clk); v_wv = 1'b1; v_wd = vw[t*NWV + i];
            end
            @(negedge clk); v_wv = 1'b0;
            @(posedge pos_done);
            repeat (2) @(negedge clk);
        end
        pos_last = 1'b0; repeat (4) @(negedge clk);

        for (hh = 0; hh < NQ; hh = hh + 1) begin
            bad = 0;
            for (i = 0; i < N; i = i + 1) begin
                @(negedge clk);
                o_rd_head = hh[0]; o_rd_addr = i[6:0];
                @(negedge clk);
                if (o_rd_data !== $signed((hh==0) ? eo0[i] : eo1[i])) begin
                    bad = bad + 1;
                    if (bad <= 3)
                        $display("Error: G%0d o[%0d] got %012h exp %012h", hh+1, i,
                                 o_rd_data, (hh==0) ? eo0[i] : eo1[i]);
                end
            end
            errors = errors + bad;
            @(negedge clk); o_rd_head = hh[0]; @(negedge clk);
            if (l_out !== ((hh==0) ? el0[0] : el1[0])) begin
                errors = errors + 1;
                $display("Error: G%0d l got %012h exp %012h", hh+1, l_out,
                         (hh==0) ? el0[0] : el1[0]);
            end
            $display("  G%0d Q head %0d: %0d/%0d coords bit-exact, l %0d, m %0d",
                     hh+1, hh, N-bad, N, l_out, m_out);
        end

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

    initial begin
        #60000000; $display("Error: timeout"); $display("FAILED"); $finish;
    end
endmodule
