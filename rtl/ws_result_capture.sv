`timescale 1ns / 1ps
//====================================================================
// ws_result_capture: debug-only deskew + store for the WS bottom-edge
// INT32 accumulator stream, so a host can read a COMPLETE result back
// over AXI-Lite. Instantiated only when DEBUG_CAPTURE != 0.
//
//   Why this exists: ws_quant_unit's bottom-edge outputs are a LIVE,
//   column-skewed streaming tap with no storage anywhere in the
//   design. Without this module there is nothing for a peek to read -
//   results only exist for the cycle they stream past. This is a
//   bring-up aid, not the data path; DMA/AXI-HP remains the real one.
//
//   Deskew: column j's values arrive on their own schedule (column j
//   lags column 0 by j cycles), so each column owns its own write
//   index. This is ws_tiling_engine's accumulate-on-write bank shape
//   minus the accumulate - one bank per column, written in arrival
//   order. Per-column write ports keep these in fabric/LUTRAM.
//
//   Arming: v_cnt[j] is cleared by `arm` (the core go pulse), so
//   every run rewrites its banks from vector 0 and a second run
//   cannot leave stale values at any index it writes. Indices beyond
//   a run's n_vecs deliberately retain the previous run's data -
//   hosts read 0..n_vecs-1.
//
//   BOTH streams are captured: the raw INT32 accumulator sums AND the
//   requantized INT8 values, so a hardware mismatch localises to the
//   array (sums wrong) or the requant lanes (sums right, INT8 wrong)
//   without another build. They arrive at DIFFERENT times - the
//   requant lanes run at ws_quant_unit's RQ_PIPE depth - so each bank
//   has its own valid and its own write index. Nothing here assumes a
//   fixed offset between the two streams, which is what lets RQ_PIPE
//   change without touching this module.
//
//   HISTORY, because it is easy to re-break: capturing INT8 from
//   COMBINATIONAL requant lanes is what caused a -3.857ns setup
//   violation (array sum_reg -> requant -> flop measured 8.355ns with
//   4.8-5.0ns of that LOGIC alone, against a 4.700ns period). It was
//   removed in a44910d and restored only once requant itself was
//   pipelined. Do not wire an UNPIPELINED requant output into a
//   synchronous consumer at this clock - see requant.sv's header for
//   the cut points and why registering its output is not enough.
//   ws_tpu_tiled_top still registers rq_final into res_q8 from a
//   PIPE_STAGES=0 instance and has never been synthesized, so it
//   carries that same latent violation.
//
//   Input pipeline stage: the streams are registered at this module's
//   boundary before the bank write, so the long haul from the array's
//   output registers to this block gets a full cycle of its own and
//   the bank write starts from a local flop. Costs one extra cycle of
//   capture latency, which is nothing on a debug path polled ~20 aclk
//   later.
//
//   Read port is REGISTERED (rd_data valid one cycle after rd_en) so
//   the wide element mux never sits on a combinational path. Zero
//   DSPs: storage and muxing only.
//====================================================================
module ws_result_capture #(
    parameter int N          = 4,
    parameter int ACC_WIDTH  = 32,
    parameter int OUT_WIDTH  = 8,
    parameter int VEC_W      = 4,
    parameter int MAX_N_VECS = 16
)(
    input  logic clk,
    input  logic rst_n,

    // clears every column's write index (drive with the core go pulse)
    input  logic arm,

    // bottom-edge INT32 stream (column-skewed, one valid per column)
    input  logic signed [ACC_WIDTH-1:0] sum_in    [N],
    input  logic                        val_in    [N],

    // requantized INT8 stream. SEPARATE valid on purpose: with the
    // requant lanes pipelined (ws_quant_unit RQ_PIPE), q8 trails its
    // INT32 source, so it gets its own write index rather than
    // borrowing the sum's. Nothing here assumes a fixed offset
    // between the two streams.
    input  logic signed [OUT_WIDTH-1:0] q8_in     [N],
    input  logic                        q8_val_in [N],

    // registered debug read: rd_data valid one cycle after rd_en
    input  logic                        rd_en,
    input  logic                        rd_is_q8,   // 0 = INT32, 1 = INT8 lane
    input  logic [VEC_W-1:0]            rd_vec,
    input  logic [3:0]                  rd_idx,     // column, or q8 lane
    output logic [31:0]                 rd_data
);

    if (ACC_WIDTH != 32) begin : gen_acc_width_guard
        initial begin
            $error("ws_result_capture: INT32 peek assumes ACC_WIDTH == 32 (got %0d)",
                   ACC_WIDTH);
            $fatal(1);
        end
    end

    // rd_vec spans the full VEC_W range, so the banks must cover it
    if (MAX_N_VECS < (1 << VEC_W)) begin : gen_depth_guard
        initial begin
            $error("ws_result_capture: MAX_N_VECS (%0d) must cover the full VEC_W=%0d range (%0d)",
                   MAX_N_VECS, VEC_W, (1 << VEC_W));
            $fatal(1);
        end
    end

    //----------------------------------------------------------------
    // Banks stored as RAW BITS (unsigned) - readback is bit-exact and
    // the host applies the interpretation, which also sidesteps the
    // unreliable signedness of unpacked-array elements here.
    // Banks are datapath (no reset); v_cnt and the pipeline valid are
    // control (async reset).
    //----------------------------------------------------------------
    logic [ACC_WIDTH-1:0] sum_bank [N][MAX_N_VECS];
    logic [VEC_W-1:0]     v_cnt    [N];

    // input pipeline stage
    logic                 p1_val [N];
    logic [ACC_WIDTH-1:0] p1_sum [N];
    logic [VEC_W-1:0]     p1_idx [N];

    // INT8 banks, with their own write index and their own input pipe
    logic [OUT_WIDTH-1:0] q8_bank [N][MAX_N_VECS];
    logic [VEC_W-1:0]     q_cnt   [N];
    logic                 q1_val  [N];
    logic [OUT_WIDTH-1:0] q1_q8   [N];
    logic [VEC_W-1:0]     q1_idx  [N];

    for (genvar j = 0; j < N; j++) begin : gen_cap
        // write index advances on the INCOMING valid; the index that
        // each value belongs to travels with it through the pipe, so
        // an arm landing between the two stages cannot misplace a
        // value still in flight.
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                v_cnt[j] <= '0;
            else if (arm)
                v_cnt[j] <= '0;
            else if (val_in[j])
                v_cnt[j] <= v_cnt[j] + 1'b1;
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) p1_val[j] <= 1'b0;
            else        p1_val[j] <= val_in[j];
        end

        always_ff @(posedge clk) begin
            p1_sum[j] <= sum_in[j];
            p1_idx[j] <= v_cnt[j];
        end

        always_ff @(posedge clk) begin
            if (p1_val[j])
                sum_bank[j][p1_idx[j]] <= p1_sum[j];
        end

        // ---- INT8, same shape, independent index ----
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                q_cnt[j] <= '0;
            else if (arm)
                q_cnt[j] <= '0;
            else if (q8_val_in[j])
                q_cnt[j] <= q_cnt[j] + 1'b1;
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) q1_val[j] <= 1'b0;
            else        q1_val[j] <= q8_val_in[j];
        end

        always_ff @(posedge clk) begin
            q1_q8[j]  <= q8_in[j];
            q1_idx[j] <= q_cnt[j];
        end

        always_ff @(posedge clk) begin
            if (q1_val[j])
                q8_bank[j][q1_idx[j]] <= q1_q8[j];
        end
    end

    //----------------------------------------------------------------
    // Registered read, bounds-checked against N: this module is built
    // at N=4 as well as N=16 (tb_axi_lite_ctrl builds ws_tpu_top at
    // N=4), where columns 4..15 do not exist.
    //----------------------------------------------------------------
    logic [31:0] sum_word;
    logic [31:0] q8_word;
    int          q8_idx;

    always_comb begin
        sum_word = 32'd0;
        if (rd_idx < N)
            sum_word = sum_bank[rd_idx][rd_vec];
    end

    // q8 lane: four packed INT8s, columns lane*4 .. lane*4+3, element
    // 0 in bits [7:0] to match the buffer-word convention. Bounds
    // checked because this module is built at N=4 as well as N=16.
    always_comb begin
        q8_word = 32'd0;
        for (int k = 0; k < 4; k++) begin
            q8_idx = rd_idx[1:0] * 4 + k;
            if (q8_idx < N)
                q8_word[k*8 +: 8] = q8_bank[q8_idx][rd_vec];
        end
    end

    always_ff @(posedge clk) begin
        if (rd_en) begin
            if (rd_is_q8) rd_data <= q8_word;
            else          rd_data <= sum_word;
        end
    end

endmodule
