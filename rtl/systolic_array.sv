`timescale 1ns / 1ps
//====================================================================
// systolic_array: N x N output-stationary mesh of pe_os
//   - a enters the LEFT edge (one stream per row), flows right
//   - b enters the TOP edge (one stream per column), flows down
//   - each PE(i,j) accumulates C[i][j] locally
//   - inputs must arrive PRE-SKEWED: row i delayed i cycles,
//     column j delayed j cycles (testbench handles this for now)
//====================================================================
module systolic_array #(
    parameter int N          = 4,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
)(
    input  logic                          clk,
    input  logic                          rst_n,
    input  logic                          clear,        // broadcast to all PEs
    input  logic                          valid_in [N], // one per row, travels with a
    input  logic signed [DATA_WIDTH-1:0]  a_in     [N], // left edge
    input  logic signed [DATA_WIDTH-1:0]  b_in     [N], // top edge
    input  logic                          drain,

    output logic signed [ACC_WIDTH-1:0]   drain_out [N],   // bottom-row buses
    output logic signed [ACC_WIDTH-1:0]   c_out    [N][N],
    output logic                          acc_valid[N][N]
);

    // Inter-PE wiring. One extra column/row of wires for the edges:
    //   a_w[i][j] = the 'a' arriving at PE(i,j) from the left
    //   b_w[i][j] = the 'b' arriving at PE(i,j) from above
    logic signed [DATA_WIDTH-1:0] a_w [N][N+1];
    logic signed [DATA_WIDTH-1:0] b_w [N+1][N];
    logic                         v_w [N][N+1];
    logic signed [ACC_WIDTH-1:0]  drain_w [N+1][N];

    // Stitch the array edges to the module ports
    for (genvar i = 0; i < N; i++) begin : gen_edge
        assign a_w[i][0] = a_in[i];      // left edge feeds row i
        assign v_w[i][0] = valid_in[i];
        assign b_w[0][i] = b_in[i];      // top edge feeds column i
        assign drain_w[0][i] = '0;       // top row's drain input: zeros
    end

    // The mesh
    for (genvar i = 0; i < N; i++) begin : gen_row
        for (genvar j = 0; j < N; j++) begin : gen_col
            pe_os #(
                .DATA_WIDTH (DATA_WIDTH),
                .ACC_WIDTH  (ACC_WIDTH)
            ) u_pe (
                .clk       (clk),
                .rst_n     (rst_n),
                .clear     (clear),
                .valid_in  (v_w[i][j]),
                .a_in      (a_w[i][j]),
                .b_in      (b_w[i][j]),
                .valid_out (v_w[i][j+1]),   // to right neighbor
                .a_out     (a_w[i][j+1]),   // to right neighbor
                .b_out     (b_w[i+1][j]),   // to bottom neighbor
                .acc_out   (c_out[i][j]),
                .acc_valid (acc_valid[i][j]),
                .drain        (drain),
                .acc_in_above (drain_w[i][j])
            );
            assign drain_w[i+1][j] = c_out[i][j];   // this PE feeds the one below
        end
    end

    for (genvar j = 0; j < N; j++) begin : gen_drain_out
        assign drain_out[j] = c_out[N-1][j];
    end

endmodule