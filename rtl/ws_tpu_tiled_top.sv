`timescale 1ns / 1ps
//====================================================================
// ws_tpu_tiled_top: chip-level top with WS tiling - axi_lite_ctrl
// -> ws_tiling_engine -> ws_quant_unit. The engine sits BETWEEN the
// control plane and the core (this is deliberately not a wrapper
// around ws_tpu_top): AXI go/busy/done describe the OUTER sweep,
// and the engine drives the w_tile_idx/a_base ports that the
// single-tile top ties to 0. ws_tpu_top remains valid and untouched
// for direct single-tile debug/use.
//
//   HOST CONTRACT (beyond axi_lite_ctrl's own header): during tiled
//   operation (K_TILES > 1), the raw sum_out/sum_valid and the
//   ws_quant_unit q8_out/q8_valid streams exposed below fire on
//   EVERY inner tile and are NOT authoritative - q8 requantizes
//   per-tile partial sums. Only the engine's res_* stream carries
//   final results. Program K_TILES/N_TILES/N_VECS/REQUANT_CFG (all
//   settle-before-go), write go, poll done; if done never arrives
//   check STATUS[2] (cfg_err - the go was rejected).
//
//   Data layout (loader contract; TB now, DMA later): weight tile
//   (n_tile, k_tile) at buffer tile index n_tile*K_TILES + k_tile;
//   activation chunk k_tile at words k_tile*n_vecs .. +n_vecs-1.
//
//   Single parameter set feeds all three instances (coupling is
//   structural); the guards below fail the build if the register-map
//   packing breaks. Pure wiring + guards: zero DSPs from this file
//   itself (u_eng carries the one final-tap requant, ~2 DSPs).
//====================================================================
module ws_tpu_tiled_top #(
    parameter int N           = 4,
    parameter int K           = 4,
    parameter int DATA_WIDTH  = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int DEPTH       = 64,
    parameter int VEC_W       = 4,
    parameter int OUT_WIDTH   = 8,
    parameter int SCALE_WIDTH = 16,
    parameter int SHIFT_WIDTH = 5,
    parameter int TILE_W      = 4,
    parameter int MAX_N_VECS  = 16
)(
    // AXI-Lite subordinate (PS domain)
    input  logic        s_axi_aclk,
    input  logic        s_axi_aresetn,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [5:0]  s_axi_awaddr,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    output logic [1:0]  s_axi_bresp,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    input  logic [5:0]  s_axi_araddr,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,

    // core domain
    input  logic                          core_clk,
    input  logic                          core_rst_n,

    // raw buffer write ports (future DMA/AXI-HP path)
    input  logic                          wr_en_w,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr_w,
    input  logic [N*DATA_WIDTH-1:0]       wr_data_w,
    input  logic                          wr_en_a,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr_a,
    input  logic [N*DATA_WIDTH-1:0]       wr_data_a,

    // perf counter clear (no AXI bit yet - see docs/perf_map.md)
    input  logic                          perf_clear,

    // per-inner-tile edges (NOT authoritative when K_TILES > 1)
    output logic signed [ACC_WIDTH-1:0]   sum_out   [N],
    output logic                          sum_valid [N],
    output logic signed [OUT_WIDTH-1:0]   q8_out    [N],
    output logic                          q8_valid  [N],

    // FINAL results: the engine's post-accumulation requant stream
    output logic                          res_valid,
    output logic signed [OUT_WIDTH-1:0]   res_q8,
    output logic [VEC_W-1:0]              res_vec,
    output logic [$clog2(N)-1:0]          res_col,
    output logic [TILE_W-1:0]             res_ntile
);

    //----------------------------------------------------------------
    // Register-map contract guards (same class as ws_tpu_top's)
    //----------------------------------------------------------------
    if (SCALE_WIDTH + SHIFT_WIDTH + 1 > 32) begin : gen_cfg_pack_guard
        initial begin
            $error("ws_tpu_tiled_top: REQUANT_CFG packing (%0d bits) does not fit one 32-bit register",
                   SCALE_WIDTH + SHIFT_WIDTH + 1);
            $fatal(1);
        end
    end

    if (VEC_W > 32 || TILE_W > 32) begin : gen_reg_width_guard
        initial begin
            $error("ws_tpu_tiled_top: VEC_W (%0d) or TILE_W (%0d) exceeds a 32-bit register",
                   VEC_W, TILE_W);
            $fatal(1);
        end
    end

    //----------------------------------------------------------------
    // Control plane <-> engine <-> core wiring. SIGNAL-DESTINATION
    // CHANGES vs ws_tpu_top (the review focus per the tiling spec):
    //   axi core_go    -> u_eng.go        (was ws.go)
    //   u_eng.busy     -> axi core_busy   (was ws.busy)
    //   axi core_n_vecs-> u_eng.n_vecs; u_eng.core_n_vecs -> ws.n_vecs
    //   u_eng drives ws.go / ws.w_tile_idx / ws.a_base
    // Everything else (requant cfg, perf, write ports) keeps its
    // ws_tpu_top wiring shape.
    //----------------------------------------------------------------
    logic                          outer_go, outer_busy, cfg_err;
    logic [VEC_W-1:0]              axi_n_vecs;
    logic [TILE_W-1:0]             axi_k_tiles, axi_n_tiles;
    logic                          core_relu_en;
    logic signed [SCALE_WIDTH-1:0] core_scale_mult;
    logic [SHIFT_WIDTH-1:0]        core_shift_amt;
    logic [3:0]                    core_perf_addr;
    logic [31:0]                   core_perf_data;

    logic                          inner_go, inner_busy;
    logic [VEC_W-1:0]              inner_n_vecs;
    logic [$clog2(DEPTH/K)-1:0]    w_tile_idx;
    logic [$clog2(DEPTH)-1:0]      a_base;

    axi_lite_ctrl #(
        .VEC_W(VEC_W), .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH),
        .TILE_W(TILE_W)
    ) u_ctrl (
        .s_axi_aclk      (s_axi_aclk),
        .s_axi_aresetn   (s_axi_aresetn),
        .s_axi_awvalid   (s_axi_awvalid),
        .s_axi_awready   (s_axi_awready),
        .s_axi_awaddr    (s_axi_awaddr),
        .s_axi_wvalid    (s_axi_wvalid),
        .s_axi_wready    (s_axi_wready),
        .s_axi_wdata     (s_axi_wdata),
        .s_axi_wstrb     (s_axi_wstrb),
        .s_axi_bvalid    (s_axi_bvalid),
        .s_axi_bready    (s_axi_bready),
        .s_axi_bresp     (s_axi_bresp),
        .s_axi_arvalid   (s_axi_arvalid),
        .s_axi_arready   (s_axi_arready),
        .s_axi_araddr    (s_axi_araddr),
        .s_axi_rvalid    (s_axi_rvalid),
        .s_axi_rready    (s_axi_rready),
        .s_axi_rdata     (s_axi_rdata),
        .s_axi_rresp     (s_axi_rresp),
        .core_clk        (core_clk),
        .core_rst_n      (core_rst_n),
        .core_go         (outer_go),
        .core_busy       (outer_busy),
        .core_n_vecs     (axi_n_vecs),
        .core_k_tiles    (axi_k_tiles),
        .core_n_tiles    (axi_n_tiles),
        .core_cfg_err    (cfg_err),
        .core_relu_en    (core_relu_en),
        .core_scale_mult (core_scale_mult),
        .core_shift_amt  (core_shift_amt),
        .core_perf_addr  (core_perf_addr),
        .core_perf_data  (core_perf_data)
    );

    ws_tiling_engine #(
        .N(N), .K(K), .ACC_WIDTH(ACC_WIDTH), .OUT_WIDTH(OUT_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .TILE_W(TILE_W),
        .MAX_N_VECS(MAX_N_VECS),
        .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH)
    ) u_eng (
        .clk          (core_clk),
        .rst_n        (core_rst_n),
        .go           (outer_go),
        .n_vecs       (axi_n_vecs),
        .k_tiles      (axi_k_tiles),
        .n_tiles      (axi_n_tiles),
        .busy         (outer_busy),
        .cfg_err      (cfg_err),
        .relu_en      (core_relu_en),
        .scale_mult   (core_scale_mult),
        .shift_amt    (core_shift_amt),
        .core_go      (inner_go),
        .core_busy    (inner_busy),
        .core_n_vecs  (inner_n_vecs),
        .w_tile_idx   (w_tile_idx),
        .a_base       (a_base),
        .sum_in       (sum_out),
        .sum_valid_in (sum_valid),
        .res_valid    (res_valid),
        .res_q8       (res_q8),
        .res_vec      (res_vec),
        .res_col      (res_col),
        .res_ntile    (res_ntile)
    );

    ws_quant_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH)
    ) u_core (
        .clk        (core_clk),
        .rst_n      (core_rst_n),
        .go         (inner_go),
        .w_tile_idx (w_tile_idx),
        .a_base     (a_base),
        .n_vecs     (inner_n_vecs),
        .busy       (inner_busy),
        .perf_clear (perf_clear),
        .perf_addr  (core_perf_addr),
        .perf_data  (core_perf_data),
        .wr_en_w    (wr_en_w),
        .wr_addr_w  (wr_addr_w),
        .wr_data_w  (wr_data_w),
        .wr_en_a    (wr_en_a),
        .wr_addr_a  (wr_addr_a),
        .wr_data_a  (wr_data_a),
        .sum_out    (sum_out),
        .sum_valid  (sum_valid),
        .q8_out     (q8_out),
        .q8_valid   (q8_valid),
        .relu_en    (core_relu_en),
        .scale_mult (core_scale_mult),
        .shift_amt  (core_shift_amt)
    );

endmodule
