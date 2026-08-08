`timescale 1ns / 1ps
//====================================================================
// recip: 1/x as a mantissa and a shift, for folding into requant.
//
//   The attention scan hands back an unnormalized accumulator o[] and
//   its normalizer l. Dividing is not a divider: requant already takes
//   scale_mult and shift_amt as RUNTIME inputs, so
//
//       out = sat8( (o * mant) >>> shift )   with mant/2^shift ~ 1/l
//
//   is exactly one requant tap and no divide in the datapath. This
//   module is the piece that produces (mant, shift).
//
//   The mantissa+shift output form is rmsnorm's, deliberately: its
//   normalizer comes out as nrm_mant/nrm_shift for the same reason,
//   that the exponent range is far too wide for one fixed-point word.
//   l = 2^31 * sum(w) over a whole scan, so its magnitude moves with
//   context length; a single format would have to be sized for the
//   longest run and would waste most of its bits on the shortest.
//
//====================================================================
// RESTORING DIVISION, NOT NEWTON-RAPHSON, AND THAT IS THE POINT
//   rmsnorm's rsqrt is Newton-Raphson because it sits in a throughput
//   path. This does not: it runs ONCE per (Q head, token), against a
//   scan that just spent ~260 cycles per cached position. So the
//   trade kv_quantize made for its vector norm applies here word for
//   word - exactness is worth more than the cycles - and it buys the
//   same thing it bought there:
//
//     - mant is EXACTLY floor(2^shift / x), so the testbench can
//       demand bit-exact agreement instead of a tolerance, and there
//       is no approximation error to characterize or re-characterize
//       when a width changes.
//     - no seed LUT, so no $readmemh, so this module cannot join the
//       list of things that silently initialise to zero when Vivado's
//       working directory differs from Icarus's (commit e5489dc).
//
//   Cost is MW+1 iterations - 16 at the default - against a
//   Newton-Raphson chain that would need a LUT read plus two
//   multiplies and would still only be approximate.
//
//====================================================================
// NORMALIZATION, AND WHY shift = e + MW
//   With e the index of x's most significant set bit, 2^e <= x <
//   2^(e+1), so 1/x lands in (2^-(e+1), 2^-e]. Choosing shift = e + MW
//   puts mant = floor(2^shift / x) in [2^(MW-1), 2^MW] - the top bit
//   always set, so every one of the MW bits carries information
//   whatever the magnitude of x.
//
//   THE UPPER END IS ATTAINABLE: x an exact power of two gives
//   mant = 2^MW, one too large to fit. It saturates to 2^MW - 1
//   instead, a relative error of 2^-MW (3.1e-5 at MW=15) on that one
//   input and zero everywhere else. That is ~13 bits below an INT8
//   LSB, so it cannot move a requantized result; it is recorded
//   because a silently saturating output is worth stating, not
//   because it matters numerically.
//
//   Widths line up with requant by construction: MW=15 fits
//   scale_mult's SCALE_WIDTH=16 SIGNED (max 32767), and shift spans
//   [MW, XW-1+MW] = [15, 62] at XW=48, which needs SHIFT_WIDTH=6.
//   That is where requant's 5->6 widening comes from - a 5-bit shift
//   caps at 31 and cannot express any of the upper half of this range.
//====================================================================
module recip #(
    parameter int XW = 48,    // input width (l is 48 bits)
    parameter int MW = 15,    // mantissa bits -> requant SCALE_WIDTH-1
    parameter int SW = 6      // shift width   -> requant SHIFT_WIDTH
)(
    input  logic clk,
    input  logic rst_n,

    input  logic          go,
    input  logic [XW-1:0] x,
    output logic          busy,
    output logic          done,

    // mant / 2**shift ~= 1/x, held until the next go
    output logic [MW-1:0] mant,
    output logic [SW-1:0] shift,

    // x was zero: mant and shift are meaningless. Sticky until the
    // next go, on the same reasoning as kv_cache_store's overflow -
    // a division by zero that quietly returned a plausible number
    // would be the worst available failure mode here.
    output logic          zero
);

    localparam int EW = $clog2(XW);

    // shift = e + MW must fit in SW bits at the largest e.
    if (XW - 1 + MW > (1 << SW) - 1) begin : gen_shift_guard
        initial begin
            $error("recip: SW (%0d) too small - shift reaches %0d",
                   SW, XW - 1 + MW);
            $fatal(1);
        end
    end

    typedef enum logic [1:0] { R_IDLE, R_NORM, R_DIV, R_DONE } state_t;
    state_t state;

    logic [XW-1:0]   x_q;
    logic [EW-1:0]   e_q;
    logic [XW-1:0]   xn;         // x normalized so bit XW-1 is set
    logic [XW:0]     rem;        // one bit wider: rem<<1 can reach 2^(XW+1)-4
    logic [MW:0]     q_acc;      // MW+1 bits - the top one is the saturating case
    logic [EW:0]     dcnt;
    logic            ge;

    assign busy = (state != R_IDLE);

    //----------------------------------------------------------------
    // Priority encode: e = index of the highest set bit of x_q.
    // Dull ascending loop rather than a tree - this runs once per
    // head and the tools flatten it anyway.
    //----------------------------------------------------------------
    logic [EW-1:0] e_c;
    integer        b;
    always_comb begin
        e_c = '0;
        for (b = 0; b < XW; b = b + 1)
            if (x_q[b]) e_c = b[EW-1:0];
    end

    assign ge = (rem >= {1'b0, xn});

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= R_IDLE;
            done  <= 1'b0;
            zero  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                R_IDLE: if (go) begin
                    state <= R_NORM;
                    zero  <= 1'b0;
                end
                R_NORM: state <= R_DIV;
                R_DIV:  if (dcnt == MW[EW:0]) state <= R_DONE;
                R_DONE: begin
                    state <= R_IDLE;
                    done  <= 1'b1;
                    if (x_q == '0) zero <= 1'b1;
                end
                default: state <= R_IDLE;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (state == R_IDLE && go) x_q <= x;
    end

    //----------------------------------------------------------------
    // The division. rem starts at 2^(XW-1) rather than 0 because every
    // partial remainder before that point is 2^j < 2^(XW-1) <= xn and
    // therefore contributes a zero quotient bit - skipping them is
    // exact, not an approximation, and turns XW+MW iterations into
    // MW+1.
    //----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (state == R_NORM) begin
            e_q   <= e_c;
            xn    <= x_q << (XW - 1 - e_c);
            rem   <= {2'b01, {(XW-1){1'b0}}};
            q_acc <= '0;
            dcnt  <= '0;
        end else if (state == R_DIV) begin
            rem   <= (ge ? (rem - {1'b0, xn}) : rem) << 1;
            q_acc <= {q_acc[MW-1:0], ge};
            dcnt  <= dcnt + 1'b1;
        end
    end

    //----------------------------------------------------------------
    // q_acc[MW] set means the quotient hit 2^MW exactly, which only
    // happens for x an exact power of two - see the header.
    //----------------------------------------------------------------
    // Computed at a fixed 32 bits and narrowed on assignment. Building
    // it as {{(SW-EW){1'b0}}, e_q} + MW would be a ZERO-WIDTH
    // replication whenever SW == EW, which is the default (6 and 6).
    logic [31:0] shift_w;
    assign shift_w = e_q + MW;

    always_ff @(posedge clk) begin
        if (state == R_DONE) begin
            mant  <= q_acc[MW] ? {MW{1'b1}} : q_acc[MW-1:0];
            shift <= shift_w[SW-1:0];
        end
    end

endmodule
