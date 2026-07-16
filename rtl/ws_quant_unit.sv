`timescale 1ns / 1ps
//====================================================================
// ws_quant_unit: ws_unit + INT8 requantized bottom edge (spec 3b).
//   Wraps ws_unit UNTOUCHED. One requant lane per column sits
//   combinationally on the live bottom-edge outputs - this is a
//   true streaming tap (sum_out/sum_valid are ports on ws_unit),
//   unlike the OS integration's buffer-read tap. q8_valid is just
//   sum_valid passed through: a quantized column value is valid
//   exactly when its INT32 source is.
//   THROUGHPUT (the spec 3b assumption check): columns overlap due
//   to the output skew - min(n_vecs, N) columns are valid in the
//   same cycle, so one lane per column is required; a single shared
//   lane would only suffice for n_vecs = 1. Measured cadence is
//   certified in tb_ws_quant_unit.
//   Config (relu_en/scale_mult/shift_amt) is per-tensor, shared by
//   all lanes; plain ports until the AXI-Lite register file exists.
//====================================================================
module ws_quant_unit #(
    parameter int N           = 4,
    parameter int K           = 4,
    parameter int DATA_WIDTH  = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int DEPTH       = 64,
    parameter int VEC_W       = 4,
    parameter int OUT_WIDTH   = 8,
    parameter int SCALE_WIDTH = 16,
    parameter int SHIFT_WIDTH = 5
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // command (pass-through)
    input  logic                          go,
    input  logic [$clog2(DEPTH/K)-1:0]    w_tile_idx,
    input  logic [$clog2(DEPTH)-1:0]      a_base,
    input  logic [VEC_W-1:0]              n_vecs,
    output logic                          busy,

    // raw write ports (pass-through)
    input  logic                          wr_en_w,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr_w,
    input  logic [31:0]                   wr_data_w,
    input  logic                          wr_en_a,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr_a,
    input  logic [31:0]                   wr_data_a,

    // raw bottom-edge results (pass-through, column-skewed)
    output logic signed [ACC_WIDTH-1:0]   sum_out   [N],
    output logic                          sum_valid [N],

    // quantized bottom edge (same cadence as sum_valid)
    output logic signed [OUT_WIDTH-1:0]   q8_out   [N],
    output logic                          q8_valid [N],

    // quantization config (per-tensor, all lanes)
    input  logic                          relu_en,
    input  logic signed [SCALE_WIDTH-1:0] scale_mult,
    input  logic [SHIFT_WIDTH-1:0]        shift_amt
);

    ws_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W)
    ) u_ws (
        .clk        (clk),
        .rst_n      (rst_n),
        .go         (go),
        .w_tile_idx (w_tile_idx),
        .a_base     (a_base),
        .n_vecs     (n_vecs),
        .busy       (busy),
        .wr_en_w    (wr_en_w),
        .wr_addr_w  (wr_addr_w),
        .wr_data_w  (wr_data_w),
        .wr_en_a    (wr_en_a),
        .wr_addr_a  (wr_addr_a),
        .wr_data_a  (wr_data_a),
        .sum_out    (sum_out),
        .sum_valid  (sum_valid)
    );

    // one requant lane per column, on the live stream
    for (genvar j = 0; j < N; j++) begin : gen_lane
        requant #(
            .ACC_WIDTH(ACC_WIDTH), .OUT_WIDTH(OUT_WIDTH),
            .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH)
        ) u_rq (
            .acc_in     (sum_out[j]),
            .relu_en    (relu_en),
            .scale_mult (scale_mult),
            .shift_amt  (shift_amt),
            .out        (q8_out[j])
        );

        assign q8_valid[j] = sum_valid[j];
    end

endmodule
