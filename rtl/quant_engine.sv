`timescale 1ns / 1ps
//====================================================================
// quant_engine: tiling_engine + INT8 requantized output read path
// (spec 3a, human-confirmed integration point).
//   Wraps tiling_engine UNTOUCHED. The output buffer keeps storing
//   raw INT32 drain rows; N requant lanes sit combinationally on
//   the read port, so every out_rd read returns BOTH the raw row
//   (out_rd_data, unchanged contract) and the quantized row
//   (out_rd8_data, lane j in bits [8j+7:8j]) one cycle after
//   out_rd_en, same latency as before.
//   Config (relu_en/scale_mult/shift_amt) is shared by all lanes
//   (per-tensor quantization) and is sampled combinationally at
//   read time - it may change between readbacks of the same result.
//   Plain ports for now; the AXI-Lite register file drives them in
//   a later phase. The inline-at-drain tap (converting as rows
//   drain, before the buffer) is deferred to the AXI work.
//====================================================================
module quant_engine #(
    parameter int N           = 4,
    parameter int K           = 4,
    parameter int DATA_WIDTH  = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int CHUNK_W     = 4,
    parameter int AB_DEPTH    = 64,
    parameter int OUT_DEPTH   = 64,
    parameter int AUTO_CLEAR  = 1,
    parameter int OUT_WIDTH   = 8,
    parameter int SCALE_WIDTH = 16,
    parameter int SHIFT_WIDTH = 5
)(
    input  logic clk,
    input  logic rst_n,

    // command (pass-through)
    input  logic                            start,
    input  logic [3:0]                      m_tiles,
    input  logic [3:0]                      n_tiles,
    input  logic [3:0]                      k_tiles,
    output logic                            done,
    output logic                            busy,

    // A/B write ports (pass-through)
    input  logic                            wr_en_a,
    input  logic [$clog2(AB_DEPTH)-1:0]     wr_addr_a,
    input  logic [31:0]                     wr_data_a,
    input  logic                            wr_en_b,
    input  logic [$clog2(AB_DEPTH)-1:0]     wr_addr_b,
    input  logic [31:0]                     wr_data_b,

    // output read port: raw INT32 row + quantized INT8 row
    input  logic                            out_rd_en,
    input  logic [$clog2(OUT_DEPTH)-1:0]    out_rd_addr,
    output logic [N*ACC_WIDTH-1:0]          out_rd_data,
    output logic [N*OUT_WIDTH-1:0]          out_rd8_data,

    // quantization config (per-tensor, all lanes)
    input  logic                            relu_en,
    input  logic signed [SCALE_WIDTH-1:0]   scale_mult,
    input  logic [SHIFT_WIDTH-1:0]          shift_amt,

    // perf counter access (pass-through)
    input  logic                            perf_clear,
    input  logic [3:0]                      perf_addr,
    output logic [31:0]                     perf_data
);

    tiling_engine #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .CHUNK_W(CHUNK_W), .AB_DEPTH(AB_DEPTH), .OUT_DEPTH(OUT_DEPTH),
        .AUTO_CLEAR(AUTO_CLEAR)
    ) u_engine (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .m_tiles     (m_tiles),
        .n_tiles     (n_tiles),
        .k_tiles     (k_tiles),
        .done        (done),
        .busy        (busy),
        .wr_en_a     (wr_en_a),
        .wr_addr_a   (wr_addr_a),
        .wr_data_a   (wr_data_a),
        .wr_en_b     (wr_en_b),
        .wr_addr_b   (wr_addr_b),
        .wr_data_b   (wr_data_b),
        .out_rd_en   (out_rd_en),
        .out_rd_addr (out_rd_addr),
        .out_rd_data (out_rd_data),
        .perf_clear  (perf_clear),
        .perf_addr   (perf_addr),
        .perf_data   (perf_data)
    );

    // one requant lane per output column, on the read path
    logic signed [ACC_WIDTH-1:0] lane_acc [N];
    logic signed [OUT_WIDTH-1:0] lane_out [N];

    for (genvar j = 0; j < N; j++) begin : gen_lane
        assign lane_acc[j] = out_rd_data[j*ACC_WIDTH +: ACC_WIDTH];

        requant #(
            .ACC_WIDTH(ACC_WIDTH), .OUT_WIDTH(OUT_WIDTH),
            .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH)
        ) u_rq (
            .acc_in     (lane_acc[j]),
            .relu_en    (relu_en),
            .scale_mult (scale_mult),
            .shift_amt  (shift_amt),
            .out        (lane_out[j])
        );

        assign out_rd8_data[j*OUT_WIDTH +: OUT_WIDTH] = lane_out[j];
    end

endmodule
