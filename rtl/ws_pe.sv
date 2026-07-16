`timescale 1ns / 1ps
//====================================================================
// ws_pe: Weight-stationary processing element
//   - w_reg holds a resident weight, loaded once per tile-swap and
//     held fixed across STREAM; the partial sum flows DOWN through
//     sum_reg every cycle (no local persistent accumulator, no
//     SETTLE/DRAIN)
//   - the vertical port pair (b_in -> sum_reg -> b_out) is retasked
//     by mode:
//       LOAD  (load_mode=1): sum_reg <= b_in - raw passthrough, the
//         column acts as a shift register carrying weight words
//         (weight value in the low DATA_WIDTH bits)
//       STREAM(load_mode=0): sum_reg <= b_in + product - accumulate
//         the local contribution onto the psum from above and pass on
//   - w_capture (broadcast, 1 cycle): w_reg <= sum_reg's low byte.
//     The controller times it so every PE's target weight sits in its
//     sum_reg on the same cycle (feed rows bottom-first: after N
//     shifts all rows align; predicted load = N+1 cycles total).
//   - 2-stage MAC discipline (as mac_unit): the product of the
//     JUST-ARRIVED a_in is registered, then added the next cycle.
//     Both the psum and the product advance one row per cycle, so
//     the OS diagonal skew (input_skew, row i delayed i) still
//     aligns. valid travels with a; sum_valid marks b_out as a
//     completed contribution (piped valid, one stage behind product).
//   - array-level result convention: bottom of column j yields
//     sum_i w[i][j]*a[i], i.e. y = W_loaded^T * a. Loaders store
//     tiles transposed to get y = W*a.
//   Reset style: control regs (valids) async reset; datapath regs
//   (w_reg, sum_reg, product, a_out) unreset - valid gates them.
//====================================================================
module ws_pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          load_mode,  // 1: LOAD_WEIGHTS, 0: STREAM
    input  logic                          w_capture,  // 1-cycle: latch weight from sum_reg
    input  logic                          valid_in,
    input  logic signed [DATA_WIDTH-1:0]  a_in,
    input  logic signed [ACC_WIDTH-1:0]   b_in,       // LOAD: weight word; STREAM: psum from above

    output logic                          valid_out,
    output logic signed [DATA_WIDTH-1:0]  a_out,
    output logic signed [ACC_WIDTH-1:0]   b_out,      // = sum_reg, to the PE below
    output logic                          sum_valid   // b_out carries a completed contribution
);

    logic signed [DATA_WIDTH-1:0]   w_reg;
    logic signed [ACC_WIDTH-1:0]    sum_reg;
    logic signed [2*DATA_WIDTH-1:0] product;
    logic                           valid_pipe;

    // Control: async reset, so the array wakes up quiet
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out  <= 1'b0;
            valid_pipe <= 1'b0;
            sum_valid  <= 1'b0;
        end else begin
            valid_out  <= valid_in;                  // travels right with a
            valid_pipe <= valid_in;                  // product-stage valid
            sum_valid  <= valid_pipe && !load_mode;  // sum-stage valid
        end
    end

    // Datapath: no reset. Priority load_mode > accumulate mirrors the
    // OS accumulator's clear > drain > accumulate mux structure.
    always_ff @(posedge clk) begin
        a_out   <= a_in;                             // 1-cycle hop right
        product <= a_in * w_reg;                     // stage 1: just-arrived a

        if (w_capture)
            w_reg <= sum_reg[DATA_WIDTH-1:0];        // same-width, no extension

        if (load_mode)
            sum_reg <= b_in;                         // weight-shift passthrough
        else if (valid_pipe)
            sum_reg <= b_in + product;               // stage 2: accumulate-and-pass
        else
            sum_reg <= b_in;                         // psum flows on, no contribution
    end

    assign b_out = sum_reg;

endmodule
