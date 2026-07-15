`timescale 1ns / 1ps
//====================================================================
// matmul_unit v2: self-sequencing NxN output-stationary matmul
//   go -> latch tile -> feed/settle/drain autonomously
//   Results exit on drain_out, labeled by result_row.
//====================================================================
module matmul_unit #(
    parameter int N          = 4,
    parameter int K          = 4,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
)(
    input  logic                          clk,
    input  logic                          rst_n,

    input  logic                          go,
    input  logic signed [DATA_WIDTH-1:0]  a_tile_in [N][K],
    input  logic signed [DATA_WIDTH-1:0]  b_tile_in [K][N],

    output logic                          busy,
    output logic                          result_valid,
    output logic [$clog2(N)-1:0]          result_row,
    output logic signed [ACC_WIDTH-1:0]   drain_out [N]
);

    // controller <-> datapath nets
    logic                         clear, drain, valid_feed;
    logic signed [DATA_WIDTH-1:0] a_feed [N];
    logic signed [DATA_WIDTH-1:0] b_feed [N];

    // skewed streams
    logic                         v_skewed [N];
    logic signed [DATA_WIDTH-1:0] a_skewed [N];
    logic signed [DATA_WIDTH-1:0] b_skewed [N];

    // internal-only leftovers (pruned at synth)
    logic                         b_valid_unused [N];
    logic                         tie_zero [N];
    logic signed [ACC_WIDTH-1:0]  c_unused [N][N];
    logic                         acc_valid_unused [N][N];

    // feed valid_feed to all rows of the a-side skew
    logic v_feed_bus [N];

    for (genvar i = 0; i < N; i++) begin : gen_tie
        assign tie_zero[i]   = 1'b0;
        assign v_feed_bus[i] = valid_feed;
    end

    controller #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH)
    ) u_ctrl (
        .clk          (clk),
        .rst_n        (rst_n),
        .go           (go),
        .a_tile_in    (a_tile_in),
        .b_tile_in    (b_tile_in),
        .busy         (busy),
        .clear        (clear),
        .drain        (drain),
        .valid_feed   (valid_feed),
        .a_feed       (a_feed),
        .b_feed       (b_feed),
        .result_valid (result_valid),
        .result_row   (result_row)
    );

    input_skew #(.N(N), .DATA_WIDTH(DATA_WIDTH)) u_skew_a (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (v_feed_bus),
        .data_in   (a_feed),
        .valid_out (v_skewed),
        .data_out  (a_skewed)
    );

    input_skew #(.N(N), .DATA_WIDTH(DATA_WIDTH)) u_skew_b (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (tie_zero),
        .data_in   (b_feed),
        .valid_out (b_valid_unused),
        .data_out  (b_skewed)
    );

    systolic_array #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) u_array (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear     (clear),
        .drain     (drain),
        .valid_in  (v_skewed),
        .a_in      (a_skewed),
        .b_in      (b_skewed),
        .c_out     (c_unused),
        .acc_valid (acc_valid_unused),
        .drain_out (drain_out)
    );

endmodule