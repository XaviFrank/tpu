`timescale 1ns / 1ps
//====================================================================
// swiglu_gate: SwiGLU's non-matmul piece.
//
//   gated_i = silu(gate_i) * up_i,   silu(x) = x / (1 + e^-x)
//
//   then requantized to INT8 for down_proj. gate_proj / up_proj /
//   down_proj themselves are ordinary matmuls and run on the existing
//   array - this module is only what sits between them.
//
//   Fully streaming, one element per cycle, no buffering: unlike
//   rmsnorm this needs no second pass, because nothing here depends on
//   a whole-vector reduction.
//
//====================================================================
// PRECISION - the format question, resolved with measurement
//====================================================================
//   gate_proj and up_proj both arrive as raw INT32 accumulator values.
//   The question was whether SiLU can be indexed off those directly or
//   needs a real dequantized value first. They are NOT symmetric:
//
//   GATE MUST BE DEQUANTIZED. SiLU is not homogeneous -
//   silu(k*x) != k*silu(x); at x=2, k=3 the two differ by 5.985 vs
//   5.285. So the table has to be addressed by the REAL value, and a
//   per-tensor scale (g_scale_mult / g_shift, the same shape requant
//   already uses) converts accumulator units into Q4.12 first.
//
//   UP DOES NOT. The gating multiply is LINEAR in up, so up's scale is
//   just a constant factor on the result and folds into the requant
//   config downstream - exactly where every other call site already
//   puts its scale. up therefore stays a raw accumulator value all the
//   way to the product.
//
//   Contrast with rmsnorm, where the residual scale cancelled out of
//   the result entirely and only eps needed converting. Here it does
//   not cancel: SiLU genuinely needs real units. Same class of
//   question, opposite answer - which is why it was measured rather
//   than assumed.
//
//   INPUT RANGE came from the existing verified pipeline
//   (forward_layer0_ffn), not from fresh calibration: over the real
//   calibration set |gate| <= 5.92 and |up| <= 3.40. The table spans
//   [-8, +8) with analytic tails outside, so the real distribution sits
//   comfortably inside with the tails never exercised in practice.
//
//   ACCURACY, checked against the REAL gate distribution rather than a
//   generic sweep: 256 bins with linear interpolation give a max
//   absolute error of 2.44e-4 on real data (3.50e-4 worst case over
//   the whole domain), against a half-LSB budget of 5.5e-3 for the
//   INT8 output. 16-22x margin.
//
//====================================================================
// FORMATS
//   gate_acc, up_acc   INT32 raw accumulators, straight from the array
//   xq                 Q4.12 signed - the dequantized gate, [-8, 8)
//   silu_q             Q4.12 signed - silu is bounded above by x, so it
//                      stays in range over this domain
//   product            silu_q * up_acc, right-shifted by prod_shift to
//                      land inside requant's 32-bit input
//   out                INT8
//
//   prod_shift is a PORT for the same reason eps_fixed is in rmsnorm:
//   it depends on the real accumulator magnitude, which is per-tensor.
//
//   requant runs at PIPE_STAGES=2 (fifth call site). Its output is
//   captured synchronously, and an unpipelined requant feeding a flop
//   is the 8.355ns chain that failed timing on ws_tpu_top.
//
//   DSP budget: 2 (gate dequant, 32x16) + 1 (interpolation, 16x8) +
//   2 (silu_q * up_acc, 16x32) + 2 (requant) = 7. Confirm against the
//   synth report rather than assuming.
//====================================================================
module swiglu_gate #(
    parameter int ACC_W   = 32,
    parameter int SILU_F  = 12,     // Q4.12
    parameter int RQ_PIPE = 2,
    // See rmsnorm.sv for why this is a parameter. Vivado override:
    //   -generic SILU_LUT_PATH=C:/dev/tpu/rtl/silu_lut.memh
    parameter SILU_LUT_PATH = "rtl/silu_lut.memh"
)(
    input  logic clk,
    input  logic rst_n,

    // input stream: one (gate, up) pair per cycle
    input  logic                    in_valid,
    input  logic signed [ACC_W-1:0] gate_acc,
    input  logic signed [ACC_W-1:0] up_acc,

    // gate dequantization -> Q4.12 real value (per-tensor)
    input  logic signed [15:0]      g_scale_mult,
    input  logic [4:0]              g_shift,

    // product down-shift into requant's 32-bit input (per-tensor)
    input  logic [5:0]              prod_shift,

    // requant config (per-tensor, same as every other call site)
    input  logic                    relu_en,
    input  logic signed [15:0]      scale_mult,
    input  logic [4:0]              shift_amt,

    output logic                    out_valid,
    output logic signed [7:0]       out_data
);

    localparam int LUT_N = 257;

    if (SILU_F != 12) begin : gen_fmt_guard
        initial begin
            $error("swiglu_gate: the LUT is generated for Q4.12; SILU_F=%0d would need a new table",
                   SILU_F);
            $fatal(1);
        end
    end

    //----------------------------------------------------------------
    // S1: dequantize gate to Q4.12, saturating at the table's edges.
    // Saturation is not a corner case to ignore - it IS the analytic
    // tail: clamping to +8 makes silu(x) = table[256] ~= x, and
    // clamping to -8 makes it ~= 0, which is what the tails converge
    // to anyway.
    //----------------------------------------------------------------
    logic signed [ACC_W+15:0] deq_wide;
    logic signed [ACC_W+15:0] deq_round;
    logic signed [ACC_W+15:0] deq_shifted;
    logic signed [15:0]       xq;
    logic signed [ACC_W-1:0]  up_s1;
    logic                     v1;

    assign deq_wide  = gate_acc * g_scale_mult;
    assign deq_round = (g_shift == 5'd0) ? deq_wide
                                         : deq_wide + (48'sd1 <<< (g_shift - 5'd1));
    assign deq_shifted = deq_round >>> g_shift;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) v1 <= 1'b0;
        else        v1 <= in_valid;
    end

    always_ff @(posedge clk) begin
        if (deq_shifted > 48'sd32767)       xq <= 16'sh7FFF;
        else if (deq_shifted < -48'sd32768) xq <= 16'sh8000;
        else                                xq <= deq_shifted[15:0];
        up_s1 <= up_acc;
    end

    //----------------------------------------------------------------
    // S2: table lookup. index = (xq + 32768) >> 8, i.e. flip the sign
    // bit and take the top byte; the low byte is the interpolation
    // fraction. Both neighbours are read, which is why the table has
    // 257 entries rather than 256.
    //----------------------------------------------------------------
    (* rom_style = "block" *) logic signed [15:0] silu_lut [LUT_N];
    initial $readmemh(SILU_LUT_PATH, silu_lut);

    logic [7:0]        idx;
    logic [8:0]        idx_p1;
    logic signed [15:0] lut_a, lut_b;
    logic [7:0]        frac_s2;
    logic signed [ACC_W-1:0] up_s2;
    logic              v2;

    // The +1 neighbour MUST be computed in 9 bits. In 8-bit
    // arithmetic idx=255 wraps to 0 and silently returns silu(-8)
    // instead of the table's 257th entry - the exact case the extra
    // entry exists to serve. Real data never reaches the top bin
    // (|gate| <= 5.92), so only a directed tail test finds this.
    assign idx    = {~xq[15], xq[14:8]};
    assign idx_p1 = {1'b0, idx} + 9'd1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) v2 <= 1'b0;
        else        v2 <= v1;
    end

    always_ff @(posedge clk) begin
        lut_a   <= silu_lut[idx];
        lut_b   <= silu_lut[idx_p1];
        frac_s2 <= xq[7:0];
        up_s2   <= up_s1;
    end

    //----------------------------------------------------------------
    // S3: linear interpolation between the two table entries.
    //----------------------------------------------------------------
    logic signed [16:0] lut_diff;
    logic signed [24:0] interp;
    logic signed [15:0] silu_q;
    logic signed [ACC_W-1:0] up_s3;
    logic               v3;

    assign lut_diff = 17'(lut_b) - 17'(lut_a);
    assign interp   = lut_diff * $signed({1'b0, frac_s2});

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) v3 <= 1'b0;
        else        v3 <= v2;
    end

    always_ff @(posedge clk) begin
        silu_q <= lut_a + 16'(interp >>> 8);
        up_s3  <= up_s2;
    end

    //----------------------------------------------------------------
    // S4: the gating multiply, then down-shift into requant's input.
    //----------------------------------------------------------------
    logic signed [ACC_W+15:0] prod;
    logic signed [31:0]       rq_in;
    logic                     v4;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) v4 <= 1'b0;
        else        v4 <= v3;
    end

    always_ff @(posedge clk) begin
        prod <= silu_q * up_s3;
    end

    // ROUND-half-up before the shift, not truncate. An arithmetic
    // right shift floors, which puts a systematic NEGATIVE bias on
    // every element whenever prod_shift > 0 - silu(-8)*1 = -11 in
    // Q4.12 floors to -1 instead of 0. requant already rounds this
    // way before its own shift, so truncating here would also be
    // inconsistent with the rest of the datapath.
    logic signed [ACC_W+15:0] prod_rnd;
    assign prod_rnd = (prod_shift == 6'd0)
                    ? prod
                    : prod + (48'sd1 <<< (prod_shift - 6'd1));
    assign rq_in = 32'(prod_rnd >>> prod_shift);

    //----------------------------------------------------------------
    // S5: shared requant core, fifth call site.
    //----------------------------------------------------------------
    logic signed [7:0] rq_out;

    requant #(
        .ACC_WIDTH(32), .OUT_WIDTH(8), .SCALE_WIDTH(16), .SHIFT_WIDTH(5),
        .PIPE_STAGES(RQ_PIPE)
    ) u_rq (
        .clk        (clk),
        .acc_in     (rq_in),
        .relu_en    (relu_en),
        .scale_mult (scale_mult),
        .shift_amt  (shift_amt),
        .out        (rq_out)
    );

    logic [1:0] rq_v;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rq_v <= 2'b00;
        else        rq_v <= {rq_v[0], v4};
    end

    assign out_valid = (RQ_PIPE == 2) ? rq_v[1] : ((RQ_PIPE == 1) ? rq_v[0] : v4);
    assign out_data  = rq_out;

endmodule
