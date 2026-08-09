`timescale 1ns / 1ps
//====================================================================
// residual_add: the INT16 residual stream, and adding a stage back
// into it. Discussed since the first layer sketch, deferred every
// time as trivial. It is not quite trivial, and this is it.
//
//     r'[i] = sat16( r[i] + ((o[i] * scale_mult) >>> shift_amt) )
//
//   r is the residual stream, INT16. o is a stage output, INT8 at its
//   OWN scale - o_proj's and down_proj's requant taps each pick their
//   own, and neither matches the residual's. So the add is not an add:
//   it is a rescale into the residual's units, and then an add.
//
//   scale_mult/shift_amt are the same mantissa+shift pair requant and
//   recip already use, for the same reason - the ratio of two
//   calibrated scales is an arbitrary real, not a power of two.
//
//====================================================================
// WHY THE RESIDUAL IS 16 BITS AND THE STAGES ARE 8
//   INT8 is right for a matmul operand: it is what the array eats and
//   what the weights are calibrated against. It is wrong for the
//   residual stream, which is ACCUMULATED - attention writes into it,
//   then the FFN writes into it again, and at 8 bits each add would
//   requantize the whole stream and compound the error twice per
//   layer, 56 times over the model.
//
//   16 bits costs one extra byte per element of on-chip storage and
//   nothing in arithmetic (the add is 16+16), and it is what rmsnorm
//   already expects: its x_rd_data is signed [IN_W-1:0] with IN_W=16.
//   So the residual stream's width was already decided by the module
//   that consumes it; this file just stops pretending otherwise.
//
//====================================================================
// SATURATION IS REAL HERE, UNLIKE q_scale
//   q_scale's clamp turned out to be unreachable and was replaced by
//   an elaboration guard. This one is NOT: r and the rescaled o are
//   both data-dependent and the sum genuinely can leave INT16 - that
//   is what an under-sized residual scale looks like. So the clamp
//   stays, with a sticky flag, and the testbench drives it on purpose.
//   A residual that wrapped would corrupt every later layer while
//   still looking like a number.
//
//====================================================================
// IN-PLACE IS SAFE, AND THAT IS WHY THE PORTS LOOK LIKE THIS
//   Two read ports in, one stream out, rather than a buffer owned
//   here: the caller wires out_* straight back to the residual
//   buffer's write port. Walking i upward with a 1-cycle read latency
//   and a registered output puts the write to address i two cycles
//   after the read of address i, so an in-place update never reads a
//   value it has already overwritten. One buffer, not two.
//====================================================================
module residual_add #(
    parameter int N  = 1024,  // elements in the residual stream
    parameter int RW = 16,    // residual width - rmsnorm's IN_W
    parameter int MW = 16,    // scale mantissa (signed, like requant's)
    parameter int SW = 6      // scale shift
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                 go,
    input  logic signed [MW-1:0] scale_mult,   // stage scale -> residual scale
    input  logic [SW-1:0]        shift_amt,
    output logic                 busy,
    output logic                 done,

    // residual in, element-addressed, 1-cycle read latency
    output logic                 r_rd_en,
    output logic [$clog2(N)-1:0] r_rd_addr,
    input  logic signed [RW-1:0] r_rd_data,

    // stage output in, same shape
    output logic                 o_rd_en,
    output logic [$clog2(N)-1:0] o_rd_addr,
    input  logic signed [7:0]    o_rd_data,

    // updated residual out
    output logic                 out_valid,
    output logic [$clog2(N)-1:0] out_idx,
    output logic signed [RW-1:0] out_data,

    // sticky: some element saturated. Cleared by go.
    output logic                 ovf
);

    localparam int AW = $clog2(N);
    localparam int PW = 9 + MW;          // signed 9-bit o x signed MW

    localparam logic [AW:0] N_LAST = N - 1;

    localparam logic signed [PW:0] R_MAX =  (1 << (RW-1)) - 1;
    localparam logic signed [PW:0] R_MIN = -(1 << (RW-1));

    typedef enum logic [1:0] { A_IDLE, A_RUN, A_TAIL } state_t;
    state_t state;

    logic [AW:0] cnt;

    assign busy      = (state != A_IDLE);
    assign r_rd_en   = (state == A_RUN);
    assign o_rd_en   = (state == A_RUN);
    assign r_rd_addr = cnt[AW-1:0];
    assign o_rd_addr = cnt[AW-1:0];

    //----------------------------------------------------------------
    // Rescale then add. Round-half-up before the shift, exactly as
    // requant and q_scale do - a different rounding rule here would
    // make the residual disagree with the stage that fed it.
    //----------------------------------------------------------------
    logic signed [8:0]     o_s;
    logic signed [PW-1:0]  prod, bias, rnd, sh;
    logic signed [PW:0]    sum;

    assign o_s  = {o_rd_data[7], o_rd_data};
    assign prod = o_s * scale_mult;

    always_comb begin
        bias = '0;
        if (shift_amt != 0) bias[shift_amt - 1'b1] = 1'b1;
    end

    assign rnd = prod + bias;
    assign sh  = rnd >>> shift_amt;
    // one bit wider than either operand, so the sum itself cannot wrap
    // before the compare below decides whether it should saturate
    assign sum = {{(PW+1-RW){r_rd_data[RW-1]}}, r_rd_data} + {sh[PW-1], sh};

    logic signed [RW-1:0] out_c;
    logic                 sat_c;

    always_comb begin
        if (sum > R_MAX) begin
            out_c = {1'b0, {(RW-1){1'b1}}};
            sat_c = 1'b1;
        end else if (sum < R_MIN) begin
            out_c = {1'b1, {(RW-1){1'b0}}};
            sat_c = 1'b1;
        end else begin
            out_c = sum[RW-1:0];
            sat_c = 1'b0;
        end
    end

    //----------------------------------------------------------------
    // One cycle of read latency, so valid/idx/data are all registered
    // once together. Registering the data without the index is the
    // off-by-one kv_dequant paid for.
    //----------------------------------------------------------------
    logic          v_q;
    logic [AW-1:0] i_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_q       <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            v_q       <= (state == A_RUN);
            out_valid <= v_q;
        end
    end

    always_ff @(posedge clk) begin
        i_q      <= cnt[AW-1:0];
        out_idx  <= i_q;
        out_data <= out_c;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                     ovf <= 1'b0;
        else if (state == A_IDLE && go) ovf <= 1'b0;
        else if (v_q && sat_c)          ovf <= 1'b1;
    end

    //----------------------------------------------------------------
    // Sequencer. done pulses in the cycle the FSM is back in A_IDLE,
    // which is also the cycle the last out_valid lands - the same
    // convention rope's rot_done and q_scale's done use, and the
    // reason both had to be added.
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= A_IDLE;
            cnt   <= '0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                A_IDLE: if (go) begin
                    state <= A_RUN;
                    cnt   <= '0;
                end
                A_RUN: if (cnt == N_LAST) begin
                    state <= A_TAIL;
                    cnt   <= '0;
                end else begin
                    cnt <= cnt + 1'b1;
                end
                A_TAIL: begin
                    state <= A_IDLE;
                    done  <= 1'b1;
                end
                default: state <= A_IDLE;
            endcase
        end
    end

endmodule
