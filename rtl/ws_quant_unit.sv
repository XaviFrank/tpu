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
//   certified in tb_ws_quant_unit (N=4) and tb_ws_n16 (N=16).
//   Word width follows ws_unit: one write word = N packed int8s.
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
    parameter int SHIFT_WIDTH = 5,
    parameter int DEBUG_CAPTURE = 1,
    parameter int RQ_PIPE     = 0
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // command (pass-through)
    input  logic                          go,
    input  logic [$clog2(DEPTH/K)-1:0]    w_tile_idx,
    input  logic [$clog2(DEPTH)-1:0]      a_base,
    input  logic [VEC_W-1:0]              n_vecs,
    output logic                          busy,
    output logic                          done,

    // perf counter access (pass-through to ws_unit)
    input  logic                          perf_clear,
    input  logic [3:0]                    perf_addr,
    output logic [31:0]                   perf_data,

    // raw write ports (pass-through)
    input  logic                          wr_en_w,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr_w,
    input  logic [N*DATA_WIDTH-1:0]       wr_data_w,
    input  logic                          wr_en_a,
    input  logic [$clog2(DEPTH)-1:0]      wr_addr_a,
    input  logic [N*DATA_WIDTH-1:0]       wr_data_a,

    // debug read port (pass-through to ws_unit's buffer read mux)
    input  logic                          dbg_rd_en,
    input  logic                          dbg_rd_sel,
    input  logic [$clog2(DEPTH)-1:0]      dbg_rd_addr,
    output logic [N*DATA_WIDTH-1:0]       dbg_rd_data,

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

    logic ctl_done;

    ws_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .DEBUG_CAPTURE(DEBUG_CAPTURE)
    ) u_ws (
        .dbg_rd_en   (dbg_rd_en),
        .dbg_rd_sel  (dbg_rd_sel),
        .dbg_rd_addr (dbg_rd_addr),
        .dbg_rd_data (dbg_rd_data),
        .clk        (clk),
        .rst_n      (rst_n),
        .go         (go),
        .w_tile_idx (w_tile_idx),
        .a_base     (a_base),
        .n_vecs     (n_vecs),
        .busy       (busy),
        .done       (ctl_done),
        .perf_clear (perf_clear),
        .perf_addr  (perf_addr),
        .perf_data  (perf_data),
        .wr_en_w    (wr_en_w),
        .wr_addr_w  (wr_addr_w),
        .wr_data_w  (wr_data_w),
        .wr_en_a    (wr_en_a),
        .wr_addr_a  (wr_addr_a),
        .wr_data_a  (wr_data_a),
        .sum_out    (sum_out),
        .sum_valid  (sum_valid)
    );

    //----------------------------------------------------------------
    // One requant lane per column, on the live stream.
    //
    // RQ_PIPE sets those lanes' PIPE_STAGES. At the default 0 the
    // lanes are combinational and q8_valid is sum_valid straight
    // through, exactly as before - which is what tb_ws_quant_unit,
    // tb_ws_n16, tb_ws_tiling_engine and ws_tpu_tiled_top all keep.
    // ws_tpu_top raises it, because its q8 has a synchronous consumer
    // (ws_result_capture) and that chain cannot close at 4.7ns
    // unpipelined - see requant.sv's header.
    //
    // The valid is delayed by exactly RQ_PIPE so each q8 value stays
    // paired with its own valid; a mismatch here would silently
    // misalign every downstream consumer rather than fail loudly.
    //----------------------------------------------------------------
    //----------------------------------------------------------------
    // done, delayed to match the requant lanes.
    //
    // ws_controller's done fires after the 2N+1 flush, i.e. after the
    // last sum_valid - but q8_valid is sum_valid delayed by RQ_PIPE,
    // so at RQ_PIPE=2 the controller's done would land TWO CYCLES
    // BEFORE the last INT8 result. A consumer that stopped on it would
    // lose the last two columns, silently. Delaying by exactly RQ_PIPE
    // puts done on the final q8_valid, which is the convention rope,
    // q_scale and residual_add all use.
    //----------------------------------------------------------------
    logic [1:0] dpipe;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) dpipe <= 2'b00;
        else        dpipe <= {dpipe[0], ctl_done};
    end

    assign done = (RQ_PIPE == 2) ? dpipe[1]
                : (RQ_PIPE == 1) ? dpipe[0] : ctl_done;

    logic [1:0] vpipe [N];

    for (genvar j = 0; j < N; j++) begin : gen_lane
        requant #(
            .ACC_WIDTH(ACC_WIDTH), .OUT_WIDTH(OUT_WIDTH),
            .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH),
            .PIPE_STAGES(RQ_PIPE)
        ) u_rq (
            .clk        (clk),
            .acc_in     (sum_out[j]),
            .relu_en    (relu_en),
            .scale_mult (scale_mult),
            .shift_amt  (shift_amt),
            .out        (q8_out[j])
        );

        // valid pipe (control: async reset). Two stages exist
        // unconditionally; RQ_PIPE selects the tap, so the RQ_PIPE=0
        // path below never reads them.
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                vpipe[j][0] <= 1'b0;
                vpipe[j][1] <= 1'b0;
            end else begin
                vpipe[j][0] <= sum_valid[j];
                vpipe[j][1] <= vpipe[j][0];
            end
        end

        if (RQ_PIPE == 0) begin : gen_v_comb
            assign q8_valid[j] = sum_valid[j];
        end else if (RQ_PIPE == 1) begin : gen_v_p1
            assign q8_valid[j] = vpipe[j][0];
        end else begin : gen_v_p2
            assign q8_valid[j] = vpipe[j][1];
        end
    end

endmodule
