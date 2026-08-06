`timescale 1ns / 1ps
//====================================================================
// tb_attn_scan - streaming attention over a REAL causal row.
//
//   102 positions from the calibration text, each RoPE'd at
//   its own position, query taken at the last token. Not a synthetic
//   row: the online-softmax rescale only fires when a later position's
//   score beats the running max, and only a genuine sequence produces
//   that pattern. tb/gen_attn_vectors.py counts the rescales and
//   refuses to emit if there are none.
//
//   TWO COMPARISONS, kept apart on purpose - the same discipline the
//   write path's round trip used:
//
//     A1/A2  the accumulator and normalizer against a BIT-EXACT model
//            of this module. Correctness. Must match exactly.
//     A3     the rescale actually fired. A scan that never rescales
//            says nothing about the rescale path, which is the whole
//            mechanic section 1 was about.
//
//   End-to-end accuracy against float attention is NOT asserted here.
//   K is INT8 now, and the row reconstructs to 19.40 dB against float
//   attention (was 4.31 dB at 2-bit K) -
//   see the commit message and docs/attention_read_design_notes.md.
//   Asserting a bound here would be measuring the write path's format
//   choice while pretending to test the scan.
//====================================================================
module tb_attn_scan;

    localparam int N   = 128;
    localparam int T   = 102;   // positions in the real row
    localparam int NWK = 32;    // K line words (plain INT8)
    localparam int NWV = 13;    // V line words (3b MSE)

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    integer errors = 0;
    integer i, t;

    logic [31:0] yq   [N];
    logic [31:0] kw   [T*NWK];
    logic [31:0] vw   [T*NWV];
    logic [47:0] exp_o [N];
    logic [47:0] exp_l [1];

    initial begin
        $readmemh("tb/attn_yq0.memh", yq);
        $readmemh("tb/attn_kw.memh", kw);
        $readmemh("tb/attn_vw.memh", vw);
        $readmemh("tb/attn_o0.memh",  exp_o);
        $readmemh("tb/attn_l0.memh",  exp_l);
    end

    logic               q_wr, prep_start, prep_done;
    logic [6:0]         q_addr;
    logic signed [31:0] q_data;
    logic               scan_start, pos_last, sbusy, pos_done, scan_done;
    logic               k_wv, v_wv;
    logic [31:0]        k_wd, v_wd;
    logic [6:0]         o_addr;
    logic signed [47:0] o_data;
    logic [47:0]        l_out;
    logic signed [15:0] m_out;

    attn_scan #(.N(N), .QBITS_K(8), .QBITS_V(3), .SSHIFT(0)) dut (
        .clk(clk), .rst_n(rst_n),
        .q_wr(q_wr), .q_addr(q_addr), .q_data(q_data),
        .prep_start(prep_start), .prep_done(prep_done),
        .scan_start(scan_start), .pos_last(pos_last),
        .busy(sbusy), .pos_done(pos_done), .scan_done(scan_done),
        .k_w_valid(k_wv), .k_w_data(k_wd),
        .v_w_valid(v_wv), .v_w_data(v_wd),
        .o_rd_addr(o_addr), .o_rd_data(o_data),
        .l_out(l_out), .m_out(m_out));

    // how many times the running max moved, watched independently of
    // the DUT's own bookkeeping
    integer resc_seen;
    logic signed [15:0] m_bef;

    task automatic feed(input integer pos);
        integer j;
        begin
            while (sbusy) @(negedge clk);
            for (j = 0; j < NWK; j = j + 1) begin
                @(negedge clk);
                k_wv = 1'b1;
                k_wd = kw[pos*NWK + j];
            end
            @(negedge clk);
            k_wv = 1'b0;
            for (j = 0; j < NWV; j = j + 1) begin
                @(negedge clk);
                v_wv = 1'b1;
                v_wd = vw[pos*NWV + j];
            end
            @(negedge clk);
            v_wv = 1'b0;
            @(posedge pos_done);
            repeat (2) @(negedge clk);
        end
    endtask

    initial begin
        q_wr = 0; prep_start = 0; scan_start = 0; pos_last = 0;
        k_wv = 0; v_wv = 0; k_wd = 0; v_wd = 0; o_addr = 0;
        resc_seen = 0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        $display("=== attn_scan: real causal row, %0d positions ===", T);

        // ---- preload the rotated, pre-scaled query ----
        for (i = 0; i < N; i = i + 1) begin
            @(negedge clk);
            q_wr   = 1'b1;
            q_addr = i[6:0];
            q_data = yq[i];
        end
        @(negedge clk);
        q_wr = 1'b0;

        @(negedge clk); prep_start = 1'b1;
        @(negedge clk); prep_start = 1'b0;
        @(posedge prep_done);
        repeat (2) @(negedge clk);

        // ---- scan ----
        for (t = 0; t < T; t = t + 1) begin
            pos_last = (t == T-1);
            m_bef    = m_out;
            feed(t);
            // position 0 sets the max rather than moving it, so it is
            // not a rescale; only later movements exercise S_RESC.
            if (t > 0 && m_out !== m_bef) resc_seen = resc_seen + 1;
        end
        pos_last = 1'b0;
        repeat (4) @(negedge clk);

        // ---- A1: the 128-wide accumulator ----
        for (i = 0; i < N; i = i + 1) begin
            @(negedge clk);
            o_addr = i[6:0];
            @(negedge clk);
            if (o_data !== $signed(exp_o[i])) begin
                errors = errors + 1;
                if (errors <= 5)
                    $display("Error: A1 o[%0d] got %012h exp %012h",
                             i, o_data, exp_o[i]);
            end
        end
        $display("  A1 accumulator   %0d/%0d coordinates bit-exact",
                 N - (errors > N ? N : errors), N);

        // ---- A2: the normalizer ----
        if (l_out !== exp_l[0]) begin
            errors = errors + 1;
            $display("Error: A2 l got %012h exp %012h", l_out, exp_l[0]);
        end else begin
            $display("  A2 normalizer    l = %0d, bit-exact", l_out);
        end

        // ---- A3: the rescale mechanic actually ran ----
        if (resc_seen == 0) begin
            errors = errors + 1;
            $display("Error: A3 running max never moved - the rescale path was never exercised");
        end else begin
            $display("  A3 rescale fired %0d time(s) over %0d positions, final m = %0d",
                     resc_seen, T, m_out);
        end

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

    initial begin
        #40000000;
        $display("Error: timeout");
        $display("FAILED");
        $finish;
    end

endmodule
