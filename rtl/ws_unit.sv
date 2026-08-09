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
//   Word width: one buffer word = one tile row = one activation
//   vector = N packed int8s (element 0 in bits [7:0]); 32 bits at
//   N=4, 128 at N=16.
//   Result convention: sum_out[j] = sum_i w[i][j]*a[i]
//   (y = W_loaded^T * a; hosts store weight tiles transposed).
//====================================================================
module ws_unit #(
    parameter int N             = 4,
    parameter int K             = 4,
    parameter int DATA_WIDTH    = 8,
    parameter int ACC_WIDTH     = 32,
    parameter int DEPTH         = 64,
    parameter int VEC_W         = 4,
    parameter int DEBUG_CAPTURE = 1
)(
    input  logic                          clk,
    input  logic                          rst_n,

    input  logic                          go,
    input  logic [$clog2(DEPTH/K)-1:0]    w_tile_idx,
    input  logic [$clog2(DEPTH)-1:0]      a_base,
    input  logic [VEC_W-1:0]              n_vecs,
    output logic                          busy,
    output logic                          done,

    // perf counter access (ws_perf_counters map - see that module)
    input  logic                          perf_clear,
    input  logic [3:0]                    perf_addr,
    output logic [31:0]                   perf_data,

    // raw write ports (one word = N packed int8s, element 0 in [7:0])
    input  logic                          wr_en_w,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr_w,
    input  logic [N*DATA_WIDTH-1:0]       wr_data_w,
    input  logic                          wr_en_a,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr_a,
    input  logic [N*DATA_WIDTH-1:0]       wr_data_a,

    // debug read port (DEBUG_CAPTURE only): shares the controller's
    // read port through the mux below - see the mux comment.
    input  logic                          dbg_rd_en,
    input  logic                          dbg_rd_sel,   // 0=weight 1=activation
    input  logic [$clog2(DEPTH)-1:0]      dbg_rd_addr,
    output logic [N*DATA_WIDTH-1:0]       dbg_rd_data,

    // bottom-edge results (continuous, column-skewed)
    output logic signed [ACC_WIDTH-1:0]   sum_out   [N],
    output logic                          sum_valid [N]
);

    localparam int WORD_WIDTH = N * DATA_WIDTH;

    // controller <-> tile-buffer nets
    logic                         rd_en_w, rd_en_a;
    logic [$clog2(DEPTH)-1:0]     rd_addr_w, rd_addr_a;
    logic [WORD_WIDTH-1:0]        rd_data_w, rd_data_a;

    //----------------------------------------------------------------
    // Debug read mux (DEBUG_CAPTURE). tile_buffer's header forbids a
    // second read port - BRAM inference is fragile - so the debug read
    // does NOT add one. It muxes in FRONT of the existing port, which
    // leaves tile_buffer untouched with exactly one rd_en/rd_addr and
    // keeps BRAM inference intact. ws_controller is untouched too.
    //   The controller wins unconditionally: a peek issued mid-run is
    // simply suppressed, so it returns stale/garbage data but CANNOT
    // disturb a run. Host contract is therefore "peek while idle".
    //   rd_data lands one cycle after rd_en, so which buffer answered
    // is remembered for one cycle to steer the return mux.
    //----------------------------------------------------------------
    logic                  buf_rd_en_w, buf_rd_en_a;
    logic [$clog2(DEPTH)-1:0] buf_rd_addr_w, buf_rd_addr_a;
    logic                  dbg_fire_w, dbg_fire_a;
    logic                  dbg_sel_q;

    if (DEBUG_CAPTURE != 0) begin : gen_dbg_rd
        assign dbg_fire_w = dbg_rd_en && !dbg_rd_sel && !rd_en_w;
        assign dbg_fire_a = dbg_rd_en &&  dbg_rd_sel && !rd_en_a;

        assign buf_rd_en_w   = rd_en_w || dbg_fire_w;
        assign buf_rd_en_a   = rd_en_a || dbg_fire_a;
        assign buf_rd_addr_w = rd_en_w ? rd_addr_w : dbg_rd_addr;
        assign buf_rd_addr_a = rd_en_a ? rd_addr_a : dbg_rd_addr;

        always_ff @(posedge clk) dbg_sel_q <= dbg_rd_sel;

        assign dbg_rd_data = dbg_sel_q ? rd_data_a : rd_data_w;
    end else begin : gen_no_dbg_rd
        assign dbg_fire_w    = 1'b0;
        assign dbg_fire_a    = 1'b0;
        assign buf_rd_en_w   = rd_en_w;
        assign buf_rd_en_a   = rd_en_a;
        assign buf_rd_addr_w = rd_addr_w;
        assign buf_rd_addr_a = rd_addr_a;
        assign dbg_sel_q     = 1'b0;
        assign dbg_rd_data   = '0;
    end

    // controller <-> datapath nets
    logic                         load_mode, w_capture, valid_feed;
    logic signed [DATA_WIDTH-1:0] a_feed [N];
    logic signed [ACC_WIDTH-1:0]  b_top  [N];

    // perf Moore flags
    logic                         in_stream, in_flush, in_idle;

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
        .rd_en   (buf_rd_en_w),
        .rd_addr (buf_rd_addr_w),
        .rd_data (rd_data_w)
    );

    tile_buffer #(
        .DEPTH(DEPTH), .WORD_WIDTH(WORD_WIDTH)
    ) u_buf_a (
        .clk     (clk),
        .wr_en   (wr_en_a),
        .wr_addr (wr_addr_a),
        .wr_data (wr_data_a),
        .rd_en   (buf_rd_en_a),
        .rd_addr (buf_rd_addr_a),
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
        .done       (done),
        .in_stream  (in_stream),
        .in_flush   (in_flush),
        .in_idle    (in_idle),
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

    ws_perf_counters u_perf (
        .clk        (clk),
        .rst_n      (rst_n),
        .perf_clear (perf_clear),
        .in_load    (load_mode),
        .in_stream  (in_stream),
        .in_flush   (in_flush),
        .in_idle    (in_idle),
        .tile_swap  (w_capture),
        .perf_addr  (perf_addr),
        .perf_data  (perf_data)
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
