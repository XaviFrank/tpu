`timescale 1ns / 1ps
//====================================================================
// requant: INT32 -> INT8 requantization with optional ReLU.
//   Generic single-value core (array-agnostic): integrations
//   instantiate one lane per tapped column (OS drain rows and WS
//   bottom-edge outputs both deliver up to N values per cycle).
//
//   Math (per-TENSOR scale; per-channel is a known future
//   refinement, deliberately not built here):
//     1. relu_out = relu_en ? max(0, acc) : acc   (pre-scale: scale
//        is always > 0, so max(0,x) commutes with the multiply)
//     2. product  = relu_out * scale_mult         (32x16 -> 48 bits)
//     3. rounded  = product + (1 << (shift_amt-1))  round-half-up;
//        NO bias when shift_amt == 0 (there is no half to round)
//     4. shifted  = rounded >>> shift_amt         (sign-preserving)
//     5. out      = saturate to [-128, 127]
//
//   Width choices (spec left to implementation, note here):
//     scale_mult : signed 16 bits. Positive multipliers in practice
//       (15 magnitude bits, ~3e-5 scale resolution); signed keeps
//       the arithmetic uniform and the DSP mapping signed-signed.
//     shift_amt  : 5 bits (0..31), covering the full useful range
//       for a 48-bit product down to int8.
//     Intermediates sized so nothing ever wraps: product 48 bits,
//       rounded/shifted 49 bits (bias can carry out of bit 47).
//
//   PURELY COMBINATIONAL (spec step 1): the 32x16 multiply is a
//   real multiplier and will be this path's critical section -
//   check the synth report's DSP column (32x16 exceeds one DSP48's
//   27x18; expect a 2-DSP cascade per lane, same inference class
//   that bit ws_pe twice). Config ports are plain inputs; the
//   AXI-Lite register file drives them in a later phase.
//====================================================================
module requant #(
    parameter int ACC_WIDTH   = 32,
    parameter int OUT_WIDTH   = 8,
    parameter int SCALE_WIDTH = 16,
    parameter int SHIFT_WIDTH = 5
)(
    input  logic signed [ACC_WIDTH-1:0]   acc_in,
    input  logic                          relu_en,
    input  logic signed [SCALE_WIDTH-1:0] scale_mult,
    input  logic [SHIFT_WIDTH-1:0]        shift_amt,
    output logic signed [OUT_WIDTH-1:0]   out
);

    localparam int PROD_W = ACC_WIDTH + SCALE_WIDTH;   // 48
    localparam int RND_W  = PROD_W + 1;                // 49: bias carry

    localparam int OUT_MAX = 2**(OUT_WIDTH-1) - 1;     //  127
    localparam int OUT_MIN = -(2**(OUT_WIDTH-1));      // -128

    logic signed [ACC_WIDTH-1:0] relu_out;
    logic signed [PROD_W-1:0]    product;
    logic signed [RND_W-1:0]     bias;
    logic signed [RND_W-1:0]     rounded;
    logic signed [RND_W-1:0]     shifted;

    always_comb begin
        // step 1: optional ReLU, pre-scale
        if (relu_en && acc_in < 0)
            relu_out = '0;
        else
            relu_out = acc_in;

        // step 2: scale
        product = relu_out * scale_mult;

        // step 3: round-to-nearest bias (none when shift_amt == 0)
        bias = '0;
        if (shift_amt != 0)
            bias[shift_amt - 1'b1] = 1'b1;
        rounded = product + bias;

        // step 4: arithmetic right shift, sign-preserving
        shifted = rounded >>> shift_amt;

        // step 5: saturate to the int8 range
        if (shifted > OUT_MAX)
            out = {1'b0, {(OUT_WIDTH-1){1'b1}}};       //  127
        else if (shifted < OUT_MIN)
            out = {1'b1, {(OUT_WIDTH-1){1'b0}}};       // -128
        else
            out = shifted[OUT_WIDTH-1:0];              // same-width
    end

endmodule
