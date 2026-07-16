`timescale 1ns / 1ps
//====================================================================
// ws_array: N x N weight-stationary mesh of ws_pe
//   - a enters the LEFT edge (one stream per row, PRE-SKEWED: row i
//     delayed i cycles - input_skew does this), flows right
//   - the vertical path enters the TOP edge: weight words during
//     LOAD (all columns in parallel, rows fed bottom-first), zeros
//     during STREAM (psum seed); psums flow DOWN and exit at the
//     bottom edge continuously - there is no drain
//   - load_mode / w_capture broadcast to all PEs (global capture:
//     bottom-first feeding aligns every row simultaneously, see
//     ws_pe header)
//   - sum_out[j]/sum_valid[j]: bottom of column j, valid whenever a
//     completed contribution sits there; column j's result for a
//     given vector appears j cycles after column 0's (natural output
//     skew, consumer deskews)
//   - result convention: sum_out[j] = sum_i w[i][j]*a[i]
//====================================================================
module ws_array #(
    parameter int N          = 4,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          load_mode,     // broadcast
    input  logic                          w_capture,     // broadcast
    input  logic                          valid_in [N],  // one per row, travels with a
    input  logic signed [DATA_WIDTH-1:0]  a_in     [N],  // left edge
    input  logic signed [ACC_WIDTH-1:0]   b_top    [N],  // top edge

    output logic signed [ACC_WIDTH-1:0]   sum_out   [N], // bottom edge psums
    output logic                          sum_valid [N]
);

    // Inter-PE wiring. One extra column/row of wires for the edges:
    //   a_w[i][j] = the 'a' arriving at PE(i,j) from the left
    //   b_w[i][j] = the vertical word arriving at PE(i,j) from above
    logic signed [DATA_WIDTH-1:0] a_w [N][N+1];
    logic                         v_w [N][N+1];
    logic signed [ACC_WIDTH-1:0]  b_w [N+1][N];
    logic                         sv_w [N][N];

    // Stitch the array edges to the module ports
    for (genvar i = 0; i < N; i++) begin : gen_edge
        assign a_w[i][0] = a_in[i];      // left edge feeds row i
        assign v_w[i][0] = valid_in[i];
        assign b_w[0][i] = b_top[i];     // top edge feeds column i
    end

    // The mesh
    for (genvar i = 0; i < N; i++) begin : gen_row
        for (genvar j = 0; j < N; j++) begin : gen_col
            ws_pe #(
                .DATA_WIDTH (DATA_WIDTH),
                .ACC_WIDTH  (ACC_WIDTH)
            ) u_pe (
                .clk       (clk),
                .rst_n     (rst_n),
                .load_mode (load_mode),
                .w_capture (w_capture),
                .valid_in  (v_w[i][j]),
                .a_in      (a_w[i][j]),
                .b_in      (b_w[i][j]),
                .valid_out (v_w[i][j+1]),   // to right neighbor
                .a_out     (a_w[i][j+1]),   // to right neighbor
                .b_out     (b_w[i+1][j]),   // to bottom neighbor
                .sum_valid (sv_w[i][j])
            );
        end
    end

    for (genvar j = 0; j < N; j++) begin : gen_out
        assign sum_out[j]   = b_w[N][j];
        assign sum_valid[j] = sv_w[N-1][j];
    end

endmodule
