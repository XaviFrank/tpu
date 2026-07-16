`timescale 1ns / 1ps
//====================================================================
// ws_unit: self-sequencing weight-stationary matvec unit (parallel
// to matmul_unit for the OS build - buffers + controller + skew +
// array under one roof).
//   Load a weight tile and activation vectors through the raw write
//   ports (TB now, DMA/AXI in Phase 3), then:
//     go + w_tile_idx + a_base + n_vecs ->
//   controller shifts the weight tile in (resident in the PEs),
//   streams n_vecs activation vectors, and the products exit
//   CONTINUOUSLY at the bottom edge - no settle, no drain.
//   sum_out[j]/sum_valid[j] are the raw bottom-edge column outputs
//   (column j lags column 0 by j cycles; consumer deskews).
//   Result convention: sum_out[j] = sum_i w[i][j]*a[i]
//   (y = W_loaded^T * a; hosts store weight tiles transposed).
//====================================================================
module ws_unit #(
    parameter int N          = 4,
    parameter int K          = 4,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32,
    parameter int DEPTH      = 64,
    parameter int VEC_W      = 4
)(
    input  logic                          clk,
    input  logic                          rst_n,

    input  logic                          go,
    input  logic [$clog2(DEPTH/K)-1:0]    w_tile_idx,
    input  logic [$clog2(DEPTH)-1:0]      a_base,
    input  logic [VEC_W-1:0]              n_vecs,
    output logic                          busy,

    // raw write ports (one word = 4 packed int8s, element 0 in [7:0])
    input  logic                          wr_en_w,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr_w,
    input  logic [31:0]                   wr_data_w,
    input  logic                          wr_en_a,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr_a,
    input  logic [31:0]                   wr_data_a,

    // bottom-edge results (continuous, column-skewed)
    output logic signed [ACC_WIDTH-1:0]   sum_out   [N],
    output logic                          sum_valid [N]
);

    localparam int WORD_WIDTH = 32;

    // controller <-> tile-buffer nets
    logic                         rd_en_w, rd_en_a;
    logic [$clog2(DEPTH)-1:0]     rd_addr_w, rd_addr_a;
    logic [WORD_WIDTH-1:0]        rd_data_w, rd_data_a;

    // controller <-> datapath nets
    logic                         load_mode, w_capture, valid_feed;
    logic signed [DATA_WIDTH-1:0] a_feed [N];
    logic signed [ACC_WIDTH-1:0]  b_top  [N];

    // skewed activation streams
    logic                         v_skewed [N];
    logic signed [DATA_WIDTH-1:0] a_skewed [N];

    // feed valid_feed to all rows of the a-side skew
    logic v_feed_bus [N];

    for (genvar i = 0; i < N; i++) begin : gen_vbus
        assign v_feed_bus[i] = valid_feed;
    end

    tile_buffer #(
        .DEPTH(DEPTH), .WORD_WIDTH(WORD_WIDTH)
    ) u_buf_w (
        .clk     (clk),
        .wr_en   (wr_en_w),
        .wr_addr (wr_addr_w),
        .wr_data (wr_data_w),
        .rd_en   (rd_en_w),
        .rd_addr (rd_addr_w),
        .rd_data (rd_data_w)
    );

    tile_buffer #(
        .DEPTH(DEPTH), .WORD_WIDTH(WORD_WIDTH)
    ) u_buf_a (
        .clk     (clk),
        .wr_en   (wr_en_a),
        .wr_addr (wr_addr_a),
        .wr_data (wr_data_a),
        .rd_en   (rd_en_a),
        .rd_addr (rd_addr_a),
        .rd_data (rd_data_a)
    );

    ws_controller #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W)
    ) u_ctrl (
        .clk        (clk),
        .rst_n      (rst_n),
        .go         (go),
        .w_tile_idx (w_tile_idx),
        .a_base     (a_base),
        .n_vecs     (n_vecs),
        .busy       (busy),
        .rd_en_w    (rd_en_w),
        .rd_addr_w  (rd_addr_w),
        .rd_data_w  (rd_data_w),
        .rd_en_a    (rd_en_a),
        .rd_addr_a  (rd_addr_a),
        .rd_data_a  (rd_data_a),
        .load_mode  (load_mode),
        .w_capture  (w_capture),
        .valid_feed (valid_feed),
        .a_feed     (a_feed),
        .b_top      (b_top)
    );

    input_skew #(.N(N), .DATA_WIDTH(DATA_WIDTH)) u_skew_a (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (v_feed_bus),
        .data_in   (a_feed),
        .valid_out (v_skewed),
        .data_out  (a_skewed)
    );

    ws_array #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) u_array (
        .clk       (clk),
        .rst_n     (rst_n),
        .load_mode (load_mode),
        .w_capture (w_capture),
        .valid_in  (v_skewed),
        .a_in      (a_skewed),
        .b_top     (b_top),
        .sum_out   (sum_out),
        .sum_valid (sum_valid)
    );

endmodule
