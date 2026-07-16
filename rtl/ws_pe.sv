`timescale 1ns / 1ps
//====================================================================
// ws_pe: Weight-stationary processing element
//   - w_reg holds a resident weight, loaded once per tile-swap and
//     held fixed across STREAM; the partial sum flows DOWN through
//     sum_reg every cycle (no local persistent accumulator, no
//     SETTLE/DRAIN)
//   - vertical path (b_in -> sum_reg -> b_out) is a STATIC
//     mult-add: sum_reg <= b_in + product, every cycle, no mode
//     mux (DSP48 fusion: product = MREG, sum_reg = PREG, b_in on
//     the C port - see the 2-DSP/PE bug trail in
//     docs/ws_design_notes.md). Mode behavior falls out of the
//     product gating:
//       LOAD : valid is low, so product is 0 and the column is a
//         pure shift register carrying weight words (low byte)
//       STREAM: product = a*w of the beat that arrived last cycle
//   - CONTRACT (tripwire below): valid_in must be low >= 1 cycle
//     before and throughout LOAD, or the shifting weights would be
//     corrupted by a live product. The controller's FLUSH state
//     (2N+1 cycles between STREAM and any LOAD) guarantees it.
//   - product gating is a mux AFTER the multiply
//     (valid ? a*w : 0), not on the operand: 0*X is X in 4-state
//     sim (w_reg is X before the first capture), and this form
//     maps to the DSP48 MREG's synchronous reset (RSTM = !valid).
//   - w_capture (broadcast, 1 cycle): w_reg <= sum_reg's low byte.
//     The controller times it so every PE's target weight sits in
//     its sum_reg on the same cycle (rows fed bottom-first).
//   - 2-stage MAC discipline (as mac_unit): product of the
//     JUST-ARRIVED a_in is registered, then added the next cycle.
//     Psum and product both advance one row per cycle, so the OS
//     diagonal skew (input_skew) still aligns. valid travels with
//     a; sum_valid marks b_out as a completed contribution.
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
    (* use_dsp = "yes" *)                            // multiplier anchor - see header
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

    // Datapath: no reset. Stage 2 is deliberately mux-free (static
    // P = C + M); all gating happens at the product register.
    always_ff @(posedge clk) begin
        a_out <= a_in;                               // 1-cycle hop right

        if (valid_in)
            product <= a_in * w_reg;                 // stage 1: just-arrived a
        else
            product <= '0;                           // DSP48 RSTM form; X-safe

        if (w_capture)
            w_reg <= sum_reg[DATA_WIDTH-1:0];        // same-width, no extension

        sum_reg <= b_in + product;                   // stage 2: accumulate-and-pass
    end

    assign b_out = sum_reg;

`ifndef SYNTHESIS
    // contract tripwire: a live product entering LOAD would corrupt
    // the weight shift (same hazard class as the OS drain/feed and
    // bank-landing tripwires)
    always @(posedge clk)
        if (rst_n && load_mode && (valid_in || valid_pipe))
            $error("ws_pe: valid activity during LOAD_WEIGHTS (valid_in=%b valid_pipe=%b)",
                   valid_in, valid_pipe);
`endif

endmodule
