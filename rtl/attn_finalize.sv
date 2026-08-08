`timescale 1ns / 1ps
//====================================================================
// attn_finalize: attn_kv_top's raw (o, l) -> one INT8 vector, ready
// for O-proj.
//
//   attn_scan deliberately stops short of a usable answer. Its output
//   is UN-NORMALIZED (o and l separately) and in the ROTATED domain,
//   because both the divide and the inverse rotation are once-per-head
//   operations that had no business inside a loop that runs once per
//   cached position. This is where they happen.
//
//     o[] --> kv_irotate --> requant --> INT8
//              (butterfly, D)   ^
//     l  --> recip --------------+  (mant, shift)
//
//====================================================================
// THREE STAGES, ONE MULTIPLIER
//   The naive reading of "divide by l, undo the rotation, requantize"
//   is a divider, a butterfly and a requant tap. Only the middle one
//   turns out to need hardware of its own:
//
//     divide by l    requant already takes scale_mult and shift_amt as
//                    RUNTIME inputs, so 1/l folds into them and the
//                    "divide" is the multiply requant was doing
//                    anyway. recip produces (mant, shift); no divider
//                    appears in the datapath.
//     divide by N    the butterfly leaves the result N times too large
//                    (H*H = N*I) and N is a power of two, so it is
//                    -log2(N) on the SAME shift. Free.
//     the 2**-15     attn_scan accumulates o at 2**16 and l at 2**31,
//                    and the V codebook is Q1.15, so o/l carries a
//                    residual 2**-15. Also a constant on that shift.
//
//   All three constants collapse into
//
//       shift_amt = shift_recip - POST_SHIFT,  POST_SHIFT = 15-log2(N)
//
//   which is 8 at N=128. That is the whole reason this module is a
//   sequencer around two existing blocks rather than new arithmetic.
//
//====================================================================
// WIDTHS - where requant's SHIFT_WIDTH 5 -> 6 comes from
//   shift_recip = e + MW with e the index of l's top bit, so it spans
//   [MW, LW-1+MW] = [15, 62] and shift_amt spans [7, 54]. A 5-bit
//   shift_amt caps at 31 and cannot express the upper half of that: on
//   the measured layer-0 row l is 31410416640, e is 34, and shift_amt
//   is 41. A 5-bit field would have silently kept the low 5 bits, 9,
//   and produced a result 2**32 too large - saturated to +-127 for
//   every coordinate, which reads as a plausible-looking vector.
//
//   RW = OW + log2(N) because the butterfly multiplies magnitudes by
//   up to N. Measured on the layer-0 row, |o| peaks at 28 bits and
//   |irot(o)| at 32, so 55 is generous - but the bound is what is
//   sized for, since a scan over a longer context grows l and o
//   together.
//
//====================================================================
// SCALE OF THE RESULT
//   The INT8 out is at V's OWN quantization scale - it is a convex
//   combination of reconstructed V vectors, so it lands in V's range
//   by construction rather than by clamping. If O-proj's calibrated
//   input scale differs, the ratio is a per-tensor constant: a power
//   of two folds into POST_SHIFT here, anything else belongs in
//   O-proj's own requant, which exists downstream regardless.
//====================================================================
module attn_finalize #(
    parameter int N          = 128,
    parameter int OW         = 48,   // attn_scan's o[] width
    parameter int LW         = 48,   // attn_scan's l width
    parameter int MW         = 15,   // recip mantissa bits
    parameter int SW         = 6,    // requant SHIFT_WIDTH
    parameter int POST_SHIFT = 8,    // 15 - log2(N)
    parameter int ROT_MODE   = 0,
    parameter ROT_SIGN_PATH  = "rtl/kv_rot_sign.memh",
    parameter ROT_PI_PATH    = "rtl/kv_rot_pi.memh"
)(
    input  logic clk,
    input  logic rst_n,

    input  logic          go,
    input  logic [LW-1:0] l_in,
    output logic          busy,
    output logic          done,

    // attn_kv_top's o read port is combinational, so the address and
    // the data are in the same cycle - no skew to absorb here.
    output logic [$clog2(N)-1:0] o_rd_addr,
    input  logic signed [OW-1:0] o_rd_data,

    output logic                 out_valid,
    output logic [$clog2(N)-1:0] out_idx,
    output logic signed [7:0]    out_data,

    // l was zero - the result is meaningless. Held until the next go.
    output logic                 zero
);

    localparam int AW = $clog2(N);
    localparam int RW = OW + AW;          // butterfly headroom

    localparam logic [AW:0] N_LAST = N - 1;

    if (MW < POST_SHIFT) begin : gen_post_guard
        initial begin
            $error("attn_finalize: POST_SHIFT (%0d) exceeds the smallest shift (%0d)",
                   POST_SHIFT, MW);
            $fatal(1);
        end
    end

    typedef enum logic [2:0] {
        F_IDLE, F_LOAD, F_ROT, F_OUT, F_DONE
    } state_t;
    state_t state;

    logic [AW:0] cnt;

    assign busy = (state != F_IDLE);

    //----------------------------------------------------------------
    // 1/l. Kicked at go and left to run underneath the load pass,
    // which is N cycles against its ~19 - so it is never the critical
    // path, and F_ROT still gates on it rather than assuming so.
    //----------------------------------------------------------------
    logic          rc_go, rc_busy, rc_done;
    logic [MW-1:0] rc_mant;
    logic [SW-1:0] rc_shift;

    assign rc_go = (state == F_IDLE) && go;

    recip #(.XW(LW), .MW(MW), .SW(SW)) u_recip (
        .clk(clk), .rst_n(rst_n),
        .go(rc_go), .x(l_in),
        .busy(rc_busy), .done(rc_done),
        .mant(rc_mant), .shift(rc_shift), .zero(zero));

    //----------------------------------------------------------------
    // Inverse rotation, shared with kv_dequant.
    //----------------------------------------------------------------
    logic               ir_in_valid, ir_run, ir_calc_done;
    logic [AW-1:0]      ir_idx;
    logic [RW-1:0]      ir_in_data, ir_rd_data;

    assign ir_in_valid = (state == F_LOAD);
    assign ir_run      = (state == F_ROT);
    assign ir_idx      = cnt[AW-1:0];
    assign ir_in_data  = {{(RW-OW){o_rd_data[OW-1]}}, o_rd_data};
    assign o_rd_addr   = cnt[AW-1:0];

    kv_irotate #(.N(N), .DW(RW), .ROT_MODE(ROT_MODE),
                 .ROT_SIGN_PATH(ROT_SIGN_PATH),
                 .ROT_PI_PATH(ROT_PI_PATH)) u_irot (
        .clk(clk), .rst_n(rst_n),
        .in_valid(ir_in_valid), .in_idx(ir_idx), .in_data(ir_in_data),
        .run(ir_run), .calc_done(ir_calc_done),
        .rd_idx(ir_idx), .rd_data(ir_rd_data));

    //----------------------------------------------------------------
    // Requant, with 1/l, 1/N and the 2**-15 all inside shift_amt.
    //----------------------------------------------------------------
    logic signed [RW-1:0] rq_acc;
    logic signed [MW:0]   rq_scale;
    logic [SW-1:0]        rq_shift;
    logic signed [7:0]    rq_out;

    assign rq_acc   = $signed(ir_rd_data);
    assign rq_scale = {1'b0, rc_mant};
    assign rq_shift = rc_shift - POST_SHIFT[SW-1:0];

    requant #(.ACC_WIDTH(RW), .OUT_WIDTH(8), .SCALE_WIDTH(MW+1),
              .SHIFT_WIDTH(SW), .PIPE_STAGES(0)) u_rq (
        .clk(clk), .acc_in(rq_acc), .relu_en(1'b0),
        .scale_mult(rq_scale), .shift_amt(rq_shift), .out(rq_out));

    //----------------------------------------------------------------
    // Output. Data is registered WITH its index and valid: requant is
    // combinational at PIPE_STAGES=0, and leaving it that way while
    // the index is registered puts the value for coordinate k out
    // labelled k-1 - the bug kv_dequant already paid for once.
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) out_valid <= 1'b0;
        else        out_valid <= (state == F_OUT);
    end

    always_ff @(posedge clk) begin
        out_data <= rq_out;
        out_idx  <= cnt[AW-1:0];
    end

    //----------------------------------------------------------------
    // Sequencer
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= F_IDLE;
            cnt   <= '0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                F_IDLE: if (go) begin
                    state <= F_LOAD;
                    cnt   <= '0;
                end
                F_LOAD: if (cnt == N_LAST) begin
                    state <= F_ROT;
                    cnt   <= '0;
                end else begin
                    cnt <= cnt + 1'b1;
                end
                // !rc_busy is guaranteed by the N-cycle load, and
                // gated on anyway - a reciprocal that had not landed
                // would scale the whole vector by a stale mantissa.
                F_ROT: if (ir_calc_done && !rc_busy) begin
                    state <= F_OUT;
                    cnt   <= '0;
                end
                F_OUT: if (cnt == N_LAST) begin
                    state <= F_DONE;
                    cnt   <= '0;
                end else begin
                    cnt <= cnt + 1'b1;
                end
                F_DONE: begin
                    state <= F_IDLE;
                    done  <= 1'b1;
                end
                default: state <= F_IDLE;
            endcase
        end
    end

endmodule
