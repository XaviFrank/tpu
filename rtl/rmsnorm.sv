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
    parameter int ACC_W    = 48,
    parameter int GAMMA_W  = 16,   // gamma format: Q2.14 SIGNED
    parameter int GAMMA_F  = 14,   // site 2's gamma reaches -1.492, so
                                   // the format must be signed, not unsigned
    parameter int OUT_FRAC = 24,   // fractional bits handed to requant
    parameter int RQ_PIPE  = 2
)(
    input  logic clk,
    input  logic rst_n,

    input  logic go,
    output logic busy,

    // x buffer read port (rd_data valid 1 cycle after rd_en)
    output logic                    x_rd_en,
    output logic [$clog2(N)-1:0]    x_rd_addr,
    input  logic signed [IN_W-1:0]  x_rd_data,

    // eps ALREADY IN mean_square UNITS - see the header. Per-site, so
    // it is supplied here rather than baked in.
    input  logic [ACC_W-1:0]        eps_fixed,

    // stage 1-2 result: mean(x^2), held until the next go
    output logic                    ms_valid,
    output logic [ACC_W-1:0]        mean_square,

    // stage 3 result: 1/sqrt(mean_square + eps) as mantissa + shift,
    //   normalizer ~= nrm_mant / 2**nrm_shift
    // Kept in that form rather than one fixed-point word because the
    // exponent range is far too wide for a single format: measured M
    // spans about 2^22..2^26 across the two layer-0 sites alone.
    output logic                    nrm_valid,
    output logic [17:0]             nrm_mant,
    output logic [5:0]              nrm_shift,

    // gamma buffer read port (Q2.14 signed, same 1-cycle latency)
    output logic                    g_rd_en,
    output logic [$clog2(N)-1:0]    g_rd_addr,
    input  logic signed [GAMMA_W-1:0] g_rd_data,

    // stage 6 requant config (per-site, same reasoning as eps_fixed)
    input  logic                    relu_en,
    input  logic signed [15:0]      scale_mult,
    input  logic [4:0]              shift_amt,

    // INT8 output stream, one element per cycle
    output logic                    out_valid,
    output logic signed [7:0]       out_data,
    output logic [$clog2(N)-1:0]    out_idx
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

    typedef enum logic [3:0] {
        S_IDLE, S_SUM, S_TAIL, S_ACC, S_R1, S_R2, S_R3, S_R4, S_DONE,
        S_P2, S_P2D
    } state_t;
    state_t state;

    logic [AW-1:0] addr;
    logic [AW-1:0] addr2;          // pass 2 element index
    logic [2:0]    drain;          // pass-2 pipeline drain counter
    logic          land;               // rd_data valid this cycle
    logic [ACC_W-1:0] acc;

    // control: async reset
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            addr        <= '0;
            addr2       <= '0;
            drain       <= 3'd5;
            land        <= 1'b0;
            ms_valid    <= 1'b0;
            nrm_valid   <= 1'b0;
        end else begin
            land <= x_rd_en;
            case (state)
                S_IDLE: begin
                    if (go) begin
                        state       <= S_SUM;
                        addr        <= '0;
                        addr2       <= '0;
                        drain       <= 3'd5;
                        ms_valid    <= 1'b0;
                        nrm_valid   <= 1'b0;
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
                    state <= S_ACC;              // final element's rd_data lands here
                end
                S_ACC: begin
                    // and its PRODUCT lands here, so acc is only final
                    // at the end of this cycle. Starting rsqrt any earlier
                    // reads an accumulator short by one element - which
                    // biases the normalizer by ~1/N and is invisible to any
                    // check that only reads mean_square at ms_valid.
                    state <= S_R1;
                end
                S_R1:   state <= S_R2;
                S_R2:   state <= S_R3;
                S_R3:   state <= S_R4;
                S_R4:   state <= S_DONE;
                S_DONE: begin
                    ms_valid  <= 1'b1;
                    nrm_valid <= 1'b1;
                    state     <= S_P2;      // straight into pass 2
                end
                S_P2: begin
                    if (addr2 == ADDR_LAST) state <= S_P2D;
                    else                    addr2 <= addr2 + 1'b1;
                end
                S_P2D: begin
                    // drain the read -> multiply -> requant pipe
                    if (drain == 3'd0) state <= S_IDLE;
                    else               drain <= drain - 1'b1;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    assign busy      = (state != S_IDLE);
    assign x_rd_en   = (state == S_SUM) || (state == S_P2);
    assign x_rd_addr = (state == S_P2) ? addr2 : addr;
    assign g_rd_en   = (state == S_P2);
    assign g_rd_addr = addr2;

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


    //----------------------------------------------------------------
    // Stage 3: normalizer = 1/sqrt(mean_square + eps).
    //
    //   M is normalised to norm = M >> sh with sh EVEN, so norm lands
    //   in [2^30, 2^32) and u = norm/2^30 lies in [1,4). Then
    //   1/sqrt(M) = (1/sqrt(u)) * 2^-((30+sh)/2), and because sh is
    //   even that trailing term is an integer shift - which is why the
    //   normalisation steps two bits at a time rather than one.
    //
    //   A 256-entry seed LUT (rsqrt_lut.memh, from tb/gen_rsqrt_lut.py)
    //   gives ~8 correct bits, then ONE Newton-Raphson step
    //   y1 = y0*(1.5 - 0.5*u*y0^2) roughly doubles that. Three
    //   multiplies, all once per vector, so they cost nothing against
    //   the 1024-wide elementwise stages.
    //----------------------------------------------------------------
    logic [ACC_W-1:0] mplus;
    assign mplus = mean_square + eps_fixed;

    logic [5:0] msb_pos;
    always_comb begin
        msb_pos = 6'd0;
        for (int b = 0; b < ACC_W; b++)
            if (mplus[b]) msb_pos = 6'(b);
    end

    logic signed [7:0] sh;
    always_comb begin
        if (msb_pos[0]) sh = $signed({2'b00, msb_pos}) - 8'sd31;  // odd MSB
        else            sh = $signed({2'b00, msb_pos}) - 8'sd30;  // even MSB
    end

    logic [31:0] norm;
    logic [6:0]  sh_abs;
    assign sh_abs = sh[7] ? 7'(-sh) : 7'(sh);
    always_comb begin
        if (sh[7]) norm = 32'(mplus << sh_abs);
        else       norm = 32'(mplus >> sh_abs);
    end

    (* rom_style = "block" *) logic [17:0] seed_lut [256];
    initial $readmemh("rtl/rsqrt_lut.memh", seed_lut);

    logic [17:0] y0;
    logic [17:0] u18;
    logic [35:0] t1;
    logic [17:0] t1_16;
    logic [35:0] t2;
    logic [19:0] t3;
    logic [37:0] y1_wide;

    always_ff @(posedge clk) begin
        if (state == S_R1) begin
            y0        <= seed_lut[norm[31:24]];
            u18       <= norm[31:14];                 // Q2.16, u in [1,4)
            nrm_shift <= 6'(17 + ((30 + sh) >>> 1));
        end
        if (state == S_R2) t1      <= y0 * y0;        // Q2.34
        if (state == S_R3) t2      <= u18 * t1_16;    // Q4.32, ~= 1.0
        if (state == S_R4) y1_wide <= y0 * t3;
    end

    // y0 is Q1.17 with value <= 1, so y0^2 is Q2.34 and occupies up to
    // bit 34 - taking [33:16] would DROP that bit and zero the Newton
    // term, silently degrading the result to 1.5*y0. Shift by 18.
    assign t1_16 = t1[35:18];                         // Q2.16
    assign t3    = 20'(3 << 15) - 20'(t2[35:16] >> 1);   // 1.5 - 0.5*u*y0^2

    // y1 = (y0*t3) >> 16, back to Q1.17. Saturate rather than wrap - an
    // overshoot would otherwise become a silently tiny normalizer.
    always_comb begin
        if (|y1_wide[37:34]) nrm_mant = 18'h3FFFF;
        else                 nrm_mant = y1_wide[33:16];
    end


    //----------------------------------------------------------------
    // Stages 4-6: pass 2. One element per cycle.
    //
    //   out_real = x_i * gamma_i / sqrt(M)
    //            = x_i * (gamma_q / 2^GAMMA_F) * (nrm_mant / 2^nrm_shift)
    //
    //   so both scalings are ONE combined shift at the end rather than
    //   two rounded steps - shifting by nrm_shift before applying gamma
    //   would throw away most of the mantissa (nrm_shift reaches 40).
    //
    //   The product is handed to requant as a fixed-point value with
    //   OUT_FRAC fractional bits, i.e. out_real * 2^OUT_FRAC, and
    //   requant's own scale_mult/shift_amt map that to INT8 at the
    //   calibrated output scale. Splitting it this way keeps the
    //   dynamic part (nrm_shift, which varies per vector) inside this
    //   module and the static part (the output scale) in requant,
    //   where every other call site already puts it.
    //
    //   requant runs at PIPE_STAGES=2 here, not 0. Its output is
    //   captured synchronously, and an unpipelined requant driving a
    //   flop is exactly the 8.355ns path that failed timing on
    //   ws_tpu_top - see requant.sv's header.
    //
    //   DSP budget for this module: 1 (stage 1 x*x) + 3 (rsqrt) +
    //   1 (x*gamma, 16x16) + 2 (that 32-bit product by the 18-bit
    //   mantissa, which exceeds one DSP48's 27x18) + 2 (requant)
    //   = 9. Confirm against the synth report rather than assuming.
    //----------------------------------------------------------------
    logic                       p2_v1, p2_v2, p2_v3;
    logic [AW-1:0]              p2_i1, p2_i2, p2_i3;
    logic signed [31:0]         m1;                    // x * gamma
    logic signed [49:0]         m2;                    // * nrm_mant
    logic signed [31:0]         acc_in;
    logic signed [7:0]          rq_out;
    logic [6:0]                 out_sh;

    // total right shift: GAMMA_F + nrm_shift, less the fractional bits
    // we deliberately keep for requant
    assign out_sh = 7'(GAMMA_F) + 7'(nrm_shift) - 7'(OUT_FRAC);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            p2_v1 <= 1'b0; p2_v2 <= 1'b0; p2_v3 <= 1'b0;
        end else begin
            p2_v1 <= (state == S_P2);
            p2_v2 <= p2_v1;
            p2_v3 <= p2_v2;
        end
    end

    always_ff @(posedge clk) begin
        p2_i1 <= addr2;
        p2_i2 <= p2_i1;
        p2_i3 <= p2_i2;
        m1    <= x_rd_data * g_rd_data;            // 16x16, 1 DSP
        m2    <= m1 * $signed({1'b0, nrm_mant});   // 32x18, 2 DSPs
    end

    assign acc_in = 32'(m2 >>> out_sh);

    requant #(
        .ACC_WIDTH(32), .OUT_WIDTH(8), .SCALE_WIDTH(16), .SHIFT_WIDTH(5),
        .PIPE_STAGES(RQ_PIPE)
    ) u_rq (
        .clk        (clk),
        .acc_in     (acc_in),
        .relu_en    (relu_en),
        .scale_mult (scale_mult),
        .shift_amt  (shift_amt),
        .out        (rq_out)
    );

    // requant adds RQ_PIPE cycles, so the valid and index travel with it
    logic [1:0]        rq_v;
    logic [AW-1:0]     rq_i0, rq_i1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rq_v <= 2'b00;
        else        rq_v <= {rq_v[0], p2_v3};
    end
    always_ff @(posedge clk) begin
        rq_i0 <= p2_i3;
        rq_i1 <= rq_i0;
    end

    assign out_valid = (RQ_PIPE == 2) ? rq_v[1] : ((RQ_PIPE == 1) ? rq_v[0] : p2_v3);
    assign out_idx   = (RQ_PIPE == 2) ? rq_i1  : ((RQ_PIPE == 1) ? rq_i0  : p2_i3);
    assign out_data  = rq_out;

endmodule
