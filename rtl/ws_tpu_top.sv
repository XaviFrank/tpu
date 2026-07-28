`timescale 1ns / 1ps
//====================================================================
// ws_tpu_top: chip-level top - the first synthesizable module that
// wires the control plane (axi_lite_ctrl, PS AXI-Lite clock domain)
// to the compute core (ws_quant_unit, core clock domain). This is
// the complete AXI-Lite-controllable accelerator, not separately-
// proven pieces; set it as the Vivado top and both clocks coexist
// for the constraints/timing.xdc clock groups to apply.
//
//   - Register map + host contract: see axi_lite_ctrl's header.
//   - Parameter coupling: both instances take VEC_W / SCALE_WIDTH /
//     SHIFT_WIDTH from THIS module's single parameter set, so the
//     pairing cannot drift by construction. The guards below
//     additionally fail the build (regress) loudly if a parameter
//     edit breaks the 32-bit register-map packing itself.
//   - Single-tile scope: w_tile_idx / a_base tied 0 (no registers in
//     the AXI map yet); first extension point for WS tiling.
//   - Raw buffer write ports and bottom-edge results are top-level
//     ports: the future DMA/AXI-HP attach points (TBs drive them
//     directly today; data movement is NOT AXI-Lite's job).
//   - Pure wiring + guards: zero registers, zero DSPs expected. A
//     DSP appearing here in synth means something got miswired into
//     logic rather than a straight port connection.
//====================================================================
module ws_tpu_top #(
    parameter int N           = 4,
    parameter int K           = 4,
    parameter int DATA_WIDTH  = 8,
    parameter int ACC_WIDTH   = 32,
    parameter int DEPTH       = 64,
    parameter int VEC_W       = 4,
    parameter int OUT_WIDTH   = 8,
    parameter int SCALE_WIDTH = 16,
    parameter int SHIFT_WIDTH = 5,
    parameter int DEBUG_CAPTURE = 1
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

    // bottom-edge results (column-skewed; future DMA path)
    output logic signed [ACC_WIDTH-1:0]   sum_out   [N],
    output logic                          sum_valid [N],
    output logic signed [OUT_WIDTH-1:0]   q8_out    [N],
    output logic                          q8_valid  [N]
);

    //----------------------------------------------------------------
    // Register-map contract guards: a parameter edit that breaks the
    // packing the AXI map assumes fails the regress build at time 0
    //----------------------------------------------------------------
    if (SCALE_WIDTH + SHIFT_WIDTH + 1 > 32) begin : gen_cfg_pack_guard
        initial begin
            $error("ws_tpu_top: REQUANT_CFG packing (%0d bits) does not fit one 32-bit register",
                   SCALE_WIDTH + SHIFT_WIDTH + 1);
            $fatal(1);
        end
    end

    if (VEC_W > 32) begin : gen_vecw_guard
        initial begin
            $error("ws_tpu_top: VEC_W (%0d) exceeds the 32-bit N_VECS register", VEC_W);
            $fatal(1);
        end
    end

    //----------------------------------------------------------------
    // Control plane <-> core wiring. Net names match axi_lite_ctrl's
    // core-side port names: TB whitebox monitors reference
    // dut.core_go / dut.core_n_vecs - do not rename.
    //----------------------------------------------------------------
    logic                          core_go, core_busy;
    logic [VEC_W-1:0]              core_n_vecs;
    logic                          core_relu_en;
    logic signed [SCALE_WIDTH-1:0] core_scale_mult;
    logic [SHIFT_WIDTH-1:0]        core_shift_amt;
    logic [3:0]                    core_perf_addr;
    logic [31:0]                   core_perf_data;

    // single-tile scope: tile index / activation base tied 0
    logic [$clog2(DEPTH/K)-1:0]    tile_zero;
    logic [$clog2(DEPTH)-1:0]      abase_zero;
    assign tile_zero  = '0;
    assign abase_zero = '0;

    // tiling-era axi_lite_ctrl ports, unused in the single-tile top
    logic       cfg_err_zero;
    logic [3:0] unused_k_tiles;
    logic [3:0] unused_n_tiles;
    assign cfg_err_zero = 1'b0;

    //----------------------------------------------------------------
    // Debug poke/peek nets (see axi_lite_ctrl's POKE/PEEK contract).
    //----------------------------------------------------------------
    localparam int WORD_W = N * DATA_WIDTH;
    localparam int BUF_AW = $clog2(DEPTH);

    logic                  core_poke_valid, core_poke_sel;
    logic [BUF_AW-1:0]     core_poke_addr;
    logic [WORD_W-1:0]     core_poke_data;
    logic [11:0]           core_peek_addr;
    logic                  core_peek_valid;
    logic [31:0]           core_peek_data;

    logic                  dbg_rd_en, dbg_rd_sel;
    logic [BUF_AW-1:0]     dbg_rd_addr;
    logic [WORD_W-1:0]     dbg_rd_data;

    axi_lite_ctrl #(
        .VEC_W(VEC_W), .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH),
        .BUF_DEPTH(DEPTH), .BUF_WORD(WORD_W)
    ) u_ctrl (
        .core_poke_valid (core_poke_valid),
        .core_poke_sel   (core_poke_sel),
        .core_poke_addr  (core_poke_addr),
        .core_poke_data  (core_poke_data),
        .core_peek_addr  (core_peek_addr),
        .core_peek_valid (core_peek_valid),
        .core_peek_data  (core_peek_data),
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
        .core_go         (core_go),
        .core_busy       (core_busy),
        .core_n_vecs     (core_n_vecs),
        .core_k_tiles    (unused_k_tiles),
        .core_n_tiles    (unused_n_tiles),
        .core_cfg_err    (cfg_err_zero),
        .core_relu_en    (core_relu_en),
        .core_scale_mult (core_scale_mult),
        .core_shift_amt  (core_shift_amt),
        .core_perf_addr  (core_perf_addr),
        .core_perf_data  (core_perf_data)
    );

    //----------------------------------------------------------------
    // Poke write mux. The top-level wr_* ports (TB today, DMA later)
    // WIN over poke, so every existing TB that drives them directly
    // behaves byte-for-byte as before. On hardware they are tied off
    // and poke is the only writer.
    //----------------------------------------------------------------
    logic              poke_en_w;
    logic              mux_wr_en_w;
    logic [BUF_AW-1:0] mux_wr_addr_w;
    logic [WORD_W-1:0] mux_wr_data_w;

    assign poke_en_w     = core_poke_valid && !core_poke_sel;
    assign mux_wr_en_w   = wr_en_w || poke_en_w;
    assign mux_wr_addr_w = wr_en_w ? wr_addr_w : core_poke_addr;
    assign mux_wr_data_w = wr_en_w ? wr_data_w : core_poke_data;

    //----------------------------------------------------------------
    // Peek read pipeline: registered on both sides so the lane mux
    // never sits on a combinational path out of the buffers. Request
    // at T, buffer rd_data at T+1, result registered at T+2;
    // axi_lite_ctrl samples at T+3. buf_sel 2/3 (result banks) return
    // 0 until the capture buffer lands.
    //----------------------------------------------------------------
    logic [1:0]  peek_lane_q, peek_sel_q;
    logic        peek_v_q;
    logic [31:0] peek_data_r;
    logic [31:0] peek_lane_val;

    assign dbg_rd_en   = core_peek_valid && !core_peek_addr[11];
    assign dbg_rd_sel  = core_peek_addr[10];
    assign dbg_rd_addr = core_peek_addr[BUF_AW-1:0];

    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) peek_v_q <= 1'b0;
        else             peek_v_q <= core_peek_valid;
    end

    always_ff @(posedge core_clk) begin
        if (core_peek_valid) begin
            peek_lane_q <= core_peek_addr[7:6];
            peek_sel_q  <= core_peek_addr[11:10];
        end
    end

    assign peek_lane_val = dbg_rd_data[peek_lane_q*32 +: 32];

    always_ff @(posedge core_clk) begin
        if (peek_v_q) begin
            if (peek_sel_q[1]) peek_data_r <= 32'd0;
            else               peek_data_r <= peek_lane_val;
        end
    end

    assign core_peek_data = peek_data_r;

    ws_quant_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH),
        .DEBUG_CAPTURE(DEBUG_CAPTURE)
    ) u_core (
        .dbg_rd_en   (dbg_rd_en),
        .dbg_rd_sel  (dbg_rd_sel),
        .dbg_rd_addr (dbg_rd_addr),
        .dbg_rd_data (dbg_rd_data),
        .clk        (core_clk),
        .rst_n      (core_rst_n),
        .go         (core_go),
        .w_tile_idx (tile_zero),
        .a_base     (abase_zero),
        .n_vecs     (core_n_vecs),
        .busy       (core_busy),
        .perf_clear (perf_clear),
        .perf_addr  (core_perf_addr),
        .perf_data  (core_perf_data),
        .wr_en_w    (mux_wr_en_w),
        .wr_addr_w  (mux_wr_addr_w),
        .wr_data_w  (mux_wr_data_w),
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
