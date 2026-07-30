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
//====================================================================
// PIPE_STAGES - OPTIONAL PIPELINING, PER INSTANCE
//====================================================================
//   PIPE_STAGES = 0 (DEFAULT) is the original purely-combinational
//   core, bit- and cycle-identical to every build before this
//   parameter existed. It is what OS's frozen quant_engine tap and
//   ws_tiling_engine's final tap use, and they must keep it.
//   requant is one module TYPE instantiated separately per call site,
//   so a nonzero value at one site cannot affect any other.
//
//   Latency = PIPE_STAGES cycles from acc_in to out. clk is UNUSED at
//   PIPE_STAGES = 0 (present in the port list unconditionally so the
//   parameter can be raised without a port-list change).
//
//   Cut points, and why these two:
//     PIPE_STAGES >= 2 : register acc_in            (isolates the route)
//     PIPE_STAGES >= 1 : register product           (isolates the DSPs)
//
//   Chosen from the measured failing path on the routed 2026-07-16
//   ws_tpu_top build (superseded report; aggregate recorded here
//   because the path no longer exists to re-query - requant is
//   currently optimized away in-system):
//     array sum_reg -> requant -> capture flop
//     8.355 ns total = logic 5.037 ns (60.3%) + route 3.318 ns (39.7%)
//     20 logic levels: DSP_A_B_DATA 1, DSP_PREADD_DATA 1,
//       DSP_MULTIPLIER 1, DSP_M_DATA 1, DSP_ALU 2, DSP_OUTPUT 2
//       (= 8 levels, the 2-DSP 32x16 cascade), CARRY8 6 (the 49-bit
//       round-add plus the two saturate compares), LUT3 1 / LUT4 1 /
//       LUT6 4 (relu mux and the variable-shift mux tree).
//
//   ONE cut is not enough. Cutting only after the multiply leaves
//   segment A = the long array->lane route (most of the 3.318 ns)
//   PLUS the whole DSP cascade, about 6 ns - still over budget. The
//   route and the DSP cascade have to end up in different segments,
//   which needs a register at the input as well.
//
//   PREDICTION for PIPE_STAGES = 2, written before the confirming
//   synthesis run (project convention):
//     A: array sum_reg -> route -> acc_s        ~2.8-3.2 ns, ~0 logic
//     B: acc_s -> relu -> DSP cascade -> prod_s ~3.2-3.6 ns, DSP-bound
//     C: prod_s -> round/shift/saturate -> out  ~2.0-2.5 ns
//   All three under both the 4.700 ns constraint and the 5.000 ns the
//   PL clock is actually configured at. If B misses, the next cut is
//   inside the cascade (register between the two DSPs); if C misses,
//   cut between the shift and the saturate.
//
//   CONFIG PORTS ARE NOT PIPELINED. relu_en / scale_mult / shift_amt
//   are sampled by whichever stage uses them, so a mid-stream config
//   change could apply scale_mult to one value and shift_amt to
//   another. That is safe here only because config is per-tensor and
//   settle-before-go (see axi_lite_ctrl's host contract) - it is
//   stable for the whole run. Do not pipeline a per-channel scale
//   this way without carrying config alongside the data.
//====================================================================
module requant #(
    parameter int ACC_WIDTH   = 32,
    parameter int OUT_WIDTH   = 8,
    parameter int SCALE_WIDTH = 16,
    parameter int SHIFT_WIDTH = 5,
    parameter int PIPE_STAGES = 0
)(
    input  logic                          clk,       // unused if PIPE_STAGES==0
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

    if (PIPE_STAGES < 0 || PIPE_STAGES > 2) begin : gen_pipe_guard
        initial begin
            $error("requant: PIPE_STAGES (%0d) must be 0, 1 or 2", PIPE_STAGES);
            $fatal(1);
        end
    end

    //----------------------------------------------------------------
    // Stage boundary 1 (PIPE_STAGES >= 2): isolate the input route
    //----------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] acc_s;

    if (PIPE_STAGES >= 2) begin : gen_acc_reg
        always_ff @(posedge clk) acc_s <= acc_in;
    end else begin : gen_acc_comb
        assign acc_s = acc_in;
    end

    //----------------------------------------------------------------
    // Front: steps 1-2, ReLU then scale (the DSP cascade)
    //----------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] relu_out;
    logic signed [PROD_W-1:0]    product;

    always_comb begin
        // step 1: optional ReLU, pre-scale
        if (relu_en && acc_s < 0)
            relu_out = '0;
        else
            relu_out = acc_s;

        // step 2: scale
        product = relu_out * scale_mult;
    end

    //----------------------------------------------------------------
    // Stage boundary 2 (PIPE_STAGES >= 1): isolate the DSP cascade
    // from the CARRY8 round/saturate tail
    //----------------------------------------------------------------
    logic signed [PROD_W-1:0] product_s;

    if (PIPE_STAGES >= 1) begin : gen_prod_reg
        always_ff @(posedge clk) product_s <= product;
    end else begin : gen_prod_comb
        assign product_s = product;
    end

    //----------------------------------------------------------------
    // Back: steps 3-5, round, arithmetic shift, saturate
    //----------------------------------------------------------------
    logic signed [RND_W-1:0] bias;
    logic signed [RND_W-1:0] rounded;
    logic signed [RND_W-1:0] shifted;

    always_comb begin
        // step 3: round-to-nearest bias (none when shift_amt == 0)
        bias = '0;
        if (shift_amt != 0)
            bias[shift_amt - 1'b1] = 1'b1;
        rounded = product_s + bias;

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
