`timescale 1ns / 1ps
//====================================================================
// q_scale: INT8 query -> attn_kv_top's pre-scaled 32-bit query.
//
//   RoPE emits INT8. attn_kv_top's q_data is signed 32-bit, because
//   attn_scan folds ALL the constant scaling into the query once per
//   head so the inner loop is a bare dot product:
//
//       yq[i] = round( q_int8[i] * G )
//       G     = s_q * s_k * log2(e) * 2**SFRAC / sqrt(d_head)
//
//   Nothing in RTL did that. It existed only as a line of numpy in
//   gen_attn_vectors.py, which is fine for a testbench that reads a
//   .memh and useless to a layer that has to run on its own. This is
//   that line, in hardware.
//
//====================================================================
// IT CANNOT BE A SHIFT, AND THAT WAS WORTH CHECKING FIRST
//   G folds two INT8 calibration scales, s_q and s_k, which are
//   absmax/127 of real tensors. Measured on Qwen3 layer 0, KV head 3:
//
//       Q head 6   G = 1890.312598...   log2(G) = 10.884409
//       Q head 7   G = 1899.062695...   log2(G) = 10.891072
//
//   Neither is within 0.11 of an integer, and they could not be - a
//   power-of-two G would require s_q*s_k to land on a power of two by
//   coincidence. So this is a real multiplier, not a shift, and the
//   only question left is how wide.
//
//====================================================================
// WHY 24 BITS OF MANTISSA
//   G arrives as (g_mult, g_shift) - the same mantissa+shift form
//   recip produces and requant consumes - and the width is set by the
//   requirement that the integer form reproduce numpy's round() on
//   REAL data, not by a rule of thumb. Measured over both heads' 128
//   coordinates:
//
//       GW=16   2/128 coordinates off by one LSB
//       GW=20   2/128 off by one
//       GW=24   0/128   <- exact
//       GW=28   0/128
//
//   The off-by-ones are ~1 part in 240000 and would not move an
//   attention score. 24 is chosen anyway because the existing
//   attn_yq/attn_o/attn_l goldens are bit-exact artifacts of a
//   verified pipeline, and eight more bits on ONE multiplier - which
//   runs 128 times per head, against a scan that runs 128 times per
//   cached POSITION - is not a cost worth trading them for.
//
//   g_mult/g_shift are INPUTS, not parameters: G depends on per-tensor
//   calibration scales, so it changes per layer and per projection.
//   Same reasoning that makes requant's scale_mult an input.
//
//====================================================================
// NO SATURATION, BECAUSE IT CANNOT HAPPEN - PROVED AT ELABORATION
//   attn_scan's dot product is sized assuming |yq| < 2**31, so the
//   first version of this module carried a clamp and a sticky ovf
//   flag. Writing the testbench for it showed the clamp is DEAD: with
//   |x| <= 128 and g_mult < 2**GW,
//
//       max |y| = 2**7 * (2**GW - 1) = 2147483520  <  2**31
//
//   at GW=24, QW=32, and the bias-then-shift cannot increase that. The
//   shift only ever makes y smaller, so no (g_mult, g_shift) a caller
//   can supply reaches the rail.
//
//   Untestable defensive logic is worse than none - it reads as
//   coverage that does not exist. It is replaced by the elaboration
//   guard below, which is strictly stronger: it fails the BUILD if a
//   width change ever makes overflow reachable, instead of catching it
//   at runtime after the fact.
//====================================================================
module q_scale #(
    parameter int N  = 128,   // head_dim
    parameter int GW = 24,    // gain mantissa width - see the header
    parameter int SW = 6,     // gain shift width
    parameter int QW = 32     // attn_kv_top's q_data width
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                 go,
    input  logic [GW-1:0]        g_mult,   // unsigned: G > 0 always
    input  logic [SW-1:0]        g_shift,
    output logic                 busy,
    output logic                 done,

    // INT8 query buffer, 1-cycle read latency (same shape as the
    // buffer kv_rotate and rope read through)
    output logic                 x_rd_en,
    output logic [$clog2(N)-1:0] x_rd_addr,
    input  logic signed [7:0]    x_rd_data,

    output logic                 y_valid,
    output logic [$clog2(N)-1:0] y_idx,
    output logic signed [QW-1:0] y_out
);

    localparam int AW = $clog2(N);
    localparam int PW = 9 + GW;        // signed 9-bit x unsigned GW

    localparam logic [AW:0] N_LAST = N - 1;

    // max |y| = 2**7 * (2**GW - 1) < 2**(GW+7), and the negative rail
    // is 2**(QW-1). Written as an exponent compare so it does not
    // itself overflow 32-bit elaboration arithmetic at large GW.
    if (GW + 8 > QW) begin : gen_width_guard
        initial begin
            $error("q_scale: GW (%0d) can overflow QW (%0d) - needs GW+8 <= QW",
                   GW, QW);
            $fatal(1);
        end
    end

    typedef enum logic [1:0] { Q_IDLE, Q_RUN, Q_TAIL } state_t;
    state_t state;

    logic [AW:0] cnt;

    assign busy      = (state != Q_IDLE);
    assign x_rd_en   = (state == Q_RUN);
    assign x_rd_addr = cnt[AW-1:0];

    //----------------------------------------------------------------
    // Multiply, round half up, shift. Same three steps requant takes,
    // and deliberately the same order: the bias goes on BEFORE the
    // shift, so a negative product rounds the same way Python's round
    // and numpy's rint do on this data.
    //----------------------------------------------------------------
    logic signed [8:0]    x_s;
    logic signed [GW:0]   g_s;
    logic signed [PW-1:0] prod, bias, rnd, sh;

    assign x_s  = {x_rd_data[7], x_rd_data};
    assign g_s  = {1'b0, g_mult};
    assign prod = x_s * g_s;

    always_comb begin
        bias = '0;
        if (g_shift != 0) bias[g_shift - 1'b1] = 1'b1;
    end

    assign rnd = prod + bias;
    assign sh  = rnd >>> g_shift;

    //----------------------------------------------------------------
    // The read has one cycle of latency, so the datapath is one cycle
    // behind the address counter - hence Q_TAIL, and hence y_idx being
    // a DELAYED copy of cnt rather than cnt itself. Registering the
    // data without registering the index the same number of times is
    // the bug kv_dequant paid for once.
    //----------------------------------------------------------------
    logic            v_q;
    logic [AW-1:0]   i_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_q     <= 1'b0;
            y_valid <= 1'b0;
        end else begin
            v_q     <= (state == Q_RUN);
            y_valid <= v_q;
        end
    end

    always_ff @(posedge clk) begin
        i_q   <= cnt[AW-1:0];
        y_idx <= i_q;
        y_out <= sh[QW-1:0];
    end

    //----------------------------------------------------------------
    // Sequencer
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= Q_IDLE;
            cnt   <= '0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                Q_IDLE: if (go) begin
                    state <= Q_RUN;
                    cnt   <= '0;
                end
                Q_RUN: if (cnt == N_LAST) begin
                    state <= Q_TAIL;
                    cnt   <= '0;
                end else begin
                    cnt <= cnt + 1'b1;
                end
                // One drain cycle, then done pulses in the cycle the
                // FSM is back in Q_IDLE - which is also the cycle the
                // LAST y_valid lands, since the datapath is two
                // registers behind the counter. Same convention as
                // rope's rot_done and kv_rotate's done: never before
                // the final beat, and never while still busy.
                Q_TAIL: begin
                    state <= Q_IDLE;
                    done  <= 1'b1;
                end
                default: state <= Q_IDLE;
            endcase
        end
    end

endmodule
