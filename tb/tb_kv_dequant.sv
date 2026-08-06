`timescale 1ns / 1ps
//====================================================================
// tb_kv_dequant - the full round trip, on real layer-0 K and V.
//
//   real INT8 x -> rotate -> quantize -> pack -> APPEND TO CACHE
//               -> read back -> unpack -> reconstruct -> inverse
//               rotate -> x_hat
//
//   Nothing is injected mid-path and nothing is bypassed: the words
//   the dequantizer sees are read out of kv_cache_store, not handed
//   over from the quantizer.
//
//   TWO COMPARISONS, AND CONFLATING THEM WOULD BE THE EASY MISTAKE:
//
//     D1-D4  x_hat against a bit-exact model of the hardware. This is
//            a CORRECTNESS check and must match exactly.
//     D5     x_hat against the ORIGINAL x. This is the ACCURACY
//            result and must NOT match exactly - 3-bit quantization is
//            lossy. A zero here would mean the testbench was measuring
//            itself rather than the quantizer.
//
//   D5's bound is set from the measurement, not before it, and is
//   stated as such: the four configurations land between 10.4 and 15.5
//   dB, so the bound is 9 dB - loose enough not to be a tripwire on
//   noise, tight enough that losing a whole codeword's worth of
//   precision would break it.
//====================================================================
module tb_kv_dequant;

    localparam int N = 128;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    always #5 clk = ~clk;

    integer errors = 0;

    logic [7:0] x_k [N];
    logic [7:0] x_v [N];
    logic [7:0] e0 [N], e1 [N], e2 [N], e3 [N];
    logic [7:0] g0 [N], g1 [N], g2 [N], g3 [N];

    initial begin
        $readmemh("tb/kv_x_k.memh", x_k);
        $readmemh("tb/kv_x_v.memh", x_v);
        $readmemh("tb/kv_rt_exp_m0_v.memh", e0);
        $readmemh("tb/kv_rt_exp_m0_k.memh", e1);
        $readmemh("tb/kv_rt_exp_m1_v.memh", e2);
        $readmemh("tb/kv_rt_exp_m1_k.memh", e3);
    end

    //----------------------------------------------------------------
    // Four chains. Names are terse on purpose - the structure repeats
    // exactly four times and long names would bury that.
    //   0: mode 0, V, QBITS 3, no QJL      2: mode 1, V
    //   1: mode 0, K, QBITS 2, QJL         3: mode 1, K
    //----------------------------------------------------------------
    logic [7:0]         xb0 [N], xb1 [N], xb2 [N], xb3 [N];
    logic               st0, st1, st2, st3;
    logic [6:0]         ra0, ra1, ra2, ra3;
    logic signed [7:0]  rd0, rd1, rd2, rd3;
    logic               yv0, yv1, yv2, yv3;
    logic signed [23:0] yd0, yd1, yd2, yd3;
    logic               wv0, wv1, wv2, wv3;
    logic [4:0]         wi0, wi1, wi2, wi3;
    logic [31:0]        wd0, wd1, wd2, wd3;
    logic               qd0, qd1, qd2, qd3;
    logic [1:0]         np0, np1, np2, np3;
    logic [31:0]        cr0, cr1, cr2, cr3;
    logic               ov0, ov1, ov2, ov3;
    logic               dv0, dv1, dv2, dv3;
    logic [6:0]         di0, di1, di2, di3;
    logic signed [7:0]  dq0, dq1, dq2, dq3;
    logic               dn0, dn1, dn2, dn3;

    always_ff @(posedge clk) rd0 <= xb0[ra0];
    always_ff @(posedge clk) rd1 <= xb1[ra1];
    always_ff @(posedge clk) rd2 <= xb2[ra2];
    always_ff @(posedge clk) rd3 <= xb3[ra3];

    always_ff @(posedge clk) if (dv0) g0[di0] <= dq0;
    always_ff @(posedge clk) if (dv1) g1[di1] <= dq1;
    always_ff @(posedge clk) if (dv2) g2[di2] <= dq2;
    always_ff @(posedge clk) if (dv3) g3[di3] <= dq3;

    // read-back sequencers: rd_en for NW cycles, w_valid delayed one
    // cycle to match the store's registered read.
    logic       rbg0, rbg1, rbg2, rbg3;
    logic       rba0, rba1, rba2, rba3;
    logic [4:0] rbc0, rbc1, rbc2, rbc3;
    logic       fv0, fv1, fv2, fv3;

    `define RBSEQ(G, A, C, F, NW)                                   \
        always_ff @(posedge clk or negedge rst_n) begin             \
            if (!rst_n) begin A <= 1'b0; C <= 5'd0; F <= 1'b0; end  \
            else begin                                              \
                F <= A;                                             \
                if (G) begin A <= 1'b1; C <= 5'd0; end              \
                else if (A) begin                                   \
                    if (C == NW-1) A <= 1'b0;                       \
                    else C <= C + 5'd1;                             \
                end                                                 \
            end                                                     \
        end

    `RBSEQ(rbg0, rba0, rbc0, fv0, 13)
    `RBSEQ(rbg1, rba1, rbc1, fv1, 14)
    `RBSEQ(rbg2, rba2, rbc2, fv2, 13)
    `RBSEQ(rbg3, rba3, rbc3, fv3, 14)

    kv_rotate #(.N(N), .ROT_MODE(0)) r0 (
        .clk(clk), .rst_n(rst_n), .start(st0), .busy(),
        .x_rd_en(), .x_rd_addr(ra0), .x_rd_data(rd0),
        .y_valid(yv0), .y_idx(), .y_out(yd0), .done());
    kv_quantize #(.N(N), .QBITS(3), .QJL_EN(0),
                  .CB_PATH("rtl/kv_cb_m0_b3.memh")) q0 (
        .clk(clk), .rst_n(rst_n), .in_valid(yv0), .in_data(yd0), .busy(),
        .w_valid(wv0), .w_idx(wi0), .w_data(wd0), .done(qd0));
    kv_cache_store #(.WORDS_PER_VEC(13), .MAX_POS(2)) c0 (
        .clk(clk), .rst_n(rst_n), .clear(1'b0), .w_valid(wv0), .w_idx(wi0),
        .w_data(wd0), .commit(qd0), .n_pos(np0), .full(), .overflow(ov0),
        .rd_en(rba0), .rd_pos(2'd0), .rd_widx(rbc0), .rd_data(cr0));
    kv_dequant #(.N(N), .QBITS(3), .QJL_EN(0), .ROT_MODE(0),
                 .INV_MUL(8388608), .INV_SHIFT(30),
                 .CB_PATH("rtl/kv_cb_m0_b3.memh")) d0 (
        .clk(clk), .rst_n(rst_n), .w_valid(fv0), .w_data(cr0), .busy(),
        .o_valid(dv0), .o_idx(di0), .o_data(dq0), .done(dn0));

    kv_rotate #(.N(N), .ROT_MODE(0)) r1 (
        .clk(clk), .rst_n(rst_n), .start(st1), .busy(),
        .x_rd_en(), .x_rd_addr(ra1), .x_rd_data(rd1),
        .y_valid(yv1), .y_idx(), .y_out(yd1), .done());
    kv_quantize #(.N(N), .QBITS(2), .QJL_EN(1),
                  .CB_PATH("rtl/kv_cb_m0_b2.memh")) q1 (
        .clk(clk), .rst_n(rst_n), .in_valid(yv1), .in_data(yd1), .busy(),
        .w_valid(wv1), .w_idx(wi1), .w_data(wd1), .done(qd1));
    kv_cache_store #(.WORDS_PER_VEC(14), .MAX_POS(2)) c1 (
        .clk(clk), .rst_n(rst_n), .clear(1'b0), .w_valid(wv1), .w_idx(wi1),
        .w_data(wd1), .commit(qd1), .n_pos(np1), .full(), .overflow(ov1),
        .rd_en(rba1), .rd_pos(2'd0), .rd_widx(rbc1), .rd_data(cr1));
    kv_dequant #(.N(N), .QBITS(2), .QJL_EN(1), .ROT_MODE(0),
                 .INV_MUL(8388608), .INV_SHIFT(30),
                 .CB_PATH("rtl/kv_cb_m0_b2.memh")) d1 (
        .clk(clk), .rst_n(rst_n), .w_valid(fv1), .w_data(cr1), .busy(),
        .o_valid(dv1), .o_idx(di1), .o_data(dq1), .done(dn1));

    kv_rotate #(.N(N), .ROT_MODE(1)) r2 (
        .clk(clk), .rst_n(rst_n), .start(st2), .busy(),
        .x_rd_en(), .x_rd_addr(ra2), .x_rd_data(rd2),
        .y_valid(yv2), .y_idx(), .y_out(yd2), .done());
    kv_quantize #(.N(N), .QBITS(3), .QJL_EN(0),
                  .CB_PATH("rtl/kv_cb_m1_b3.memh")) q2 (
        .clk(clk), .rst_n(rst_n), .in_valid(yv2), .in_data(yd2), .busy(),
        .w_valid(wv2), .w_idx(wi2), .w_data(wd2), .done(qd2));
    kv_cache_store #(.WORDS_PER_VEC(13), .MAX_POS(2)) c2 (
        .clk(clk), .rst_n(rst_n), .clear(1'b0), .w_valid(wv2), .w_idx(wi2),
        .w_data(wd2), .commit(qd2), .n_pos(np2), .full(), .overflow(ov2),
        .rd_en(rba2), .rd_pos(2'd0), .rd_widx(rbc2), .rd_data(cr2));
    kv_dequant #(.N(N), .QBITS(3), .QJL_EN(0), .ROT_MODE(1),
                 .INV_MUL(9680), .INV_SHIFT(30),
                 .CB_PATH("rtl/kv_cb_m1_b3.memh")) d2 (
        .clk(clk), .rst_n(rst_n), .w_valid(fv2), .w_data(cr2), .busy(),
        .o_valid(dv2), .o_idx(di2), .o_data(dq2), .done(dn2));

    kv_rotate #(.N(N), .ROT_MODE(1)) r3 (
        .clk(clk), .rst_n(rst_n), .start(st3), .busy(),
        .x_rd_en(), .x_rd_addr(ra3), .x_rd_data(rd3),
        .y_valid(yv3), .y_idx(), .y_out(yd3), .done());
    kv_quantize #(.N(N), .QBITS(2), .QJL_EN(1),
                  .CB_PATH("rtl/kv_cb_m1_b2.memh")) q3 (
        .clk(clk), .rst_n(rst_n), .in_valid(yv3), .in_data(yd3), .busy(),
        .w_valid(wv3), .w_idx(wi3), .w_data(wd3), .done(qd3));
    kv_cache_store #(.WORDS_PER_VEC(14), .MAX_POS(2)) c3 (
        .clk(clk), .rst_n(rst_n), .clear(1'b0), .w_valid(wv3), .w_idx(wi3),
        .w_data(wd3), .commit(qd3), .n_pos(np3), .full(), .overflow(ov3),
        .rd_en(rba3), .rd_pos(2'd0), .rd_widx(rbc3), .rd_data(cr3));
    kv_dequant #(.N(N), .QBITS(2), .QJL_EN(1), .ROT_MODE(1),
                 .INV_MUL(9680), .INV_SHIFT(30),
                 .CB_PATH("rtl/kv_cb_m1_b2.memh")) d3 (
        .clk(clk), .rst_n(rst_n), .w_valid(fv3), .w_data(cr3), .busy(),
        .o_valid(dv3), .o_idx(di3), .o_data(dq3), .done(dn3));

    //----------------------------------------------------------------
    function automatic longint s8(input logic [7:0] v);
        longint t;
        begin
            t = v;
            if (v[7]) t = t - 64'd256;
            s8 = t;
        end
    endfunction

    integer i, bad;
    longint se, sx, dx;
    real    snr;

    task automatic check(input [63:0] tag, input integer sel);
        logic [7:0] gv, ev, xv;
        begin
            bad = 0; se = 0; sx = 0;
            for (i = 0; i < N; i = i + 1) begin
                case (sel)
                    0: begin gv = g0[i]; ev = e0[i]; xv = x_v[i]; end
                    1: begin gv = g1[i]; ev = e1[i]; xv = x_k[i]; end
                    2: begin gv = g2[i]; ev = e2[i]; xv = x_v[i]; end
                    default: begin gv = g3[i]; ev = e3[i]; xv = x_k[i]; end
                endcase
                if (gv !== ev) begin
                    bad = bad + 1;
                    if (bad <= 4)
                        $display("Error: %0s idx %0d got %02h exp %02h",
                                 tag, i, gv, ev);
                end
                dx = s8(xv) - s8(gv);
                se = se + dx * dx;
                sx = sx + s8(xv) * s8(xv);
            end
            errors = errors + bad;
            if (se == 0) begin
                errors = errors + 1;
                snr = 999.0;
                $display("Error: %0s round trip is LOSSLESS, which 3-bit quantization cannot be - the TB is measuring itself", tag);
            end else begin
                snr = 10.0 * $log10((sx * 1.0) / (se * 1.0));
            end
            $display("  %0s  %0d/%0d bit-exact vs model   round-trip SNR %0.2f dB",
                     tag, N - bad, N, snr);
            if (se != 0 && snr < 9.0) begin
                errors = errors + 1;
                $display("Error: %0s round-trip SNR %0.2f below 9 dB bound",
                         tag, snr);
            end
        end
    endtask

    initial begin
        st0 = 0; st1 = 0; st2 = 0; st3 = 0;
        rbg0 = 0; rbg1 = 0; rbg2 = 0; rbg3 = 0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);
        for (i = 0; i < N; i = i + 1) begin
            xb0[i] = x_v[i]; xb1[i] = x_k[i];
            xb2[i] = x_v[i]; xb3[i] = x_k[i];
        end

        $display("=== kv_dequant: full round trip on real layer-0 K/V ===");

        @(negedge clk); st0 = 1; st1 = 1; st2 = 1; st3 = 1;
        @(negedge clk); st0 = 0; st1 = 0; st2 = 0; st3 = 0;

        // mode 0 chains finish first; read each back once stored
        @(posedge qd0); repeat (2) @(negedge clk);
        rbg0 = 1; @(negedge clk); rbg0 = 0;
        @(posedge qd1); repeat (2) @(negedge clk);
        rbg1 = 1; @(negedge clk); rbg1 = 0;
        // the last coordinate is registered on the SAME edge done
        // rises, so it is captured one edge later - check after it lands
        @(posedge dn0); @(posedge dn1); repeat (2) @(negedge clk);

        $display("D1/D2 randomized Hadamard");
        check("D1 m0 V", 0);
        check("D2 m0 K", 1);

        @(posedge qd2); repeat (2) @(negedge clk);
        rbg2 = 1; @(negedge clk); rbg2 = 0;
        @(posedge qd3); repeat (2) @(negedge clk);
        rbg3 = 1; @(negedge clk); rbg3 = 0;
        @(posedge dn2); @(posedge dn3); repeat (2) @(negedge clk);

        $display("D3/D4 dense Pi");
        check("D3 m1 V", 2);
        check("D4 m1 K", 3);

        if (ov0 || ov1 || ov2 || ov3) begin
            errors = errors + 1;
            $display("Error: a cache overflowed during the round trip");
        end

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

    initial begin
        #8000000;
        $display("Error: timeout");
        $display("FAILED");
        $finish;
    end

endmodule
