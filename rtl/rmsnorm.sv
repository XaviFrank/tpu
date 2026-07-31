`timescale 1ns / 1ps
//====================================================================
// rmsnorm: RMS normalization over an N-element vector.
//
//   out_i = x_i * (1/sqrt(mean(x^2) + eps)) * gamma_i, then requantized
//   to INT8 through the shared requant core.
//
//   FIRST NON-MATMUL UNIT in this design, and the first to touch the
//   residual stream.
//
//====================================================================
// DATA FORMATS - three now, keep the boundaries explicit
//====================================================================
//   INT8   matmul operands and results (weights, activations, and this
//          module's own output after stage 6)
//   INT16  RESIDUAL STREAM - new with this module. This module's INPUT.
//   INT32  raw matmul accumulator output (ws_unit's sum_out)
//   INT48  internal only: the sum-of-squares accumulator here. Sized to
//          the DSP48 P-register width, matching the width class already
//          used elsewhere. Worst case is N * (2^15)^2 = 2^10 * 2^30 =
//          2^40, so 48 bits leaves 8 bits of headroom and cannot
//          overflow at any N <= 2^17.
//
//====================================================================
// eps IS A PORT, NOT A CONSTANT - deliberate, see below
//====================================================================
//   eps is 1e-6 in float, but this datapath needs it in units of
//   mean_square, i.e. in (INT16 LSB)^2. That value depends entirely on
//   the INT16 residual-stream scale, and measured on real Qwen3-0.6B
//   layer-0 data that scale CANNOT be a single number:
//
//     site                                input absmax   eps_fixed
//     input_layernorm  (input = embedding)     0.140         55054
//     post_attn_layernorm (input = residual)   3.277           100
//
//   a 23.5x spread in input range and a 550x spread in eps, within one
//   layer - and the residual keeps growing across the other 27. So
//   eps_fixed is supplied per invocation rather than baked in, exactly
//   as requant.sv takes scale_mult/shift_amt as ports rather than
//   hard-coding a quantization decision. Whoever fixes the residual
//   format supplies the number; this module stays correct either way.
//
//   Same reasoning for gamma (per-channel, read from a buffer) and for
//   the stage-6 requant scale/shift (ports). The output distribution
//   IS already calibrated, incidentally: RMSNorm's output is the matmul
//   input the software pipeline measured - absmax 3.194 at site 1 and
//   10.025 at site 2.
//
//====================================================================
// STRUCTURE - two passes over x, so x must be RE-READABLE
//====================================================================
//   Pass 1 accumulates sum(x^2); pass 2 rescales every element. That
//   needs x twice, so this module drives a READ PORT into an external
//   buffer rather than consuming a one-shot stream. Same shape as
//   ws_controller reading tile_buffer, including the 1-cycle read
//   latency (rd_data valid the cycle after rd_en).
//
//   THIS COMMIT implements stages 1-2 only (sum of squares, then the
//   free >>log2(N) mean). Stages 3-6 - rsqrt, the two elementwise
//   scales, and the requant tap - land in later steps.
//
//   DSP: one multiply per element in pass 1. use_dsp is placed on the
//   PRODUCT SIGNAL, not module scope - module scope is what pushed the
//   32-bit add into a second DSP during the ws_pe iterations (see
//   docs/ws_design_notes.md). Expect exactly 1 DSP for this stage;
//   confirm in the synth report rather than assuming.
//====================================================================
module rmsnorm #(
    parameter int N     = 1024,
    parameter int IN_W  = 16,
    parameter int ACC_W = 48
)(
    input  logic clk,
    input  logic rst_n,

    input  logic go,
    output logic busy,

    // x buffer read port (rd_data valid 1 cycle after rd_en)
    output logic                    x_rd_en,
    output logic [$clog2(N)-1:0]    x_rd_addr,
    input  logic signed [IN_W-1:0]  x_rd_data,

    // stage 1-2 result: mean(x^2), held until the next go
    output logic                    ms_valid,
    output logic [ACC_W-1:0]        mean_square
);

    localparam int AW    = $clog2(N);
    localparam logic [$clog2(N)-1:0] ADDR_LAST = N - 1;
    localparam int LOG2N = $clog2(N);

    if ((1 << LOG2N) != N) begin : gen_pow2_guard
        initial begin
            $error("rmsnorm: N (%0d) must be a power of two - stage 2 is a shift", N);
            $fatal(1);
        end
    end

    // 2*IN_W bits for the product, plus log2(N) for the sum, must fit
    if (2*IN_W + LOG2N > ACC_W) begin : gen_acc_guard
        initial begin
            $error("rmsnorm: ACC_W (%0d) too small for N=%0d at IN_W=%0d (needs %0d)",
                   ACC_W, N, IN_W, 2*IN_W + LOG2N);
            $fatal(1);
        end
    end

    typedef enum logic [1:0] { S_IDLE, S_SUM, S_TAIL, S_DONE } state_t;
    state_t state;

    logic [AW-1:0] addr;
    logic          land;               // rd_data valid this cycle
    logic [ACC_W-1:0] acc;

    // control: async reset
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            addr        <= '0;
            land        <= 1'b0;
            ms_valid    <= 1'b0;
        end else begin
            land <= x_rd_en;
            case (state)
                S_IDLE: begin
                    if (go) begin
                        state       <= S_SUM;
                        addr        <= '0;
                        ms_valid    <= 1'b0;
                    end
                end
                S_SUM: begin
                    if (addr == ADDR_LAST) begin
                        state       <= S_TAIL;   // one more cycle for the last read
                    end else begin
                        addr <= addr + 1'b1;
                    end
                end
                S_TAIL: begin
                    state <= S_DONE;             // the final element lands here
                end
                S_DONE: begin
                    ms_valid <= 1'b1;
                    state    <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    assign busy      = (state != S_IDLE);
    assign x_rd_en   = (state == S_SUM);
    assign x_rd_addr = addr;

    // datapath: no reset. The accumulator is cleared on go instead,
    // which also covers post-reset X (nothing reads it before ms_valid).
    (* use_dsp = "yes" *) logic signed [2*IN_W-1:0] prod;
    logic [ACC_W-1:0] prod_ext;

    always_ff @(posedge clk) begin
        prod <= x_rd_data * x_rd_data;      // always >= 0
    end

    assign prod_ext = {{(ACC_W-2*IN_W){1'b0}}, prod};

    logic land_q;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) land_q <= 1'b0;
        else        land_q <= land;         // prod is one cycle behind land
    end

    always_ff @(posedge clk) begin
        if (state == S_IDLE && go) acc <= '0;
        else if (land_q)           acc <= acc + prod_ext;
    end

    // stage 2: mean = acc >> log2(N). Free.
    assign mean_square = acc >> LOG2N;

endmodule
