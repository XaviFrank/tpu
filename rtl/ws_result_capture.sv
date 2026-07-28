`timescale 1ns / 1ps
//====================================================================
// ws_result_capture: debug-only deskew + store for the WS bottom-edge
// output streams, so a host can read a COMPLETE result back over
// AXI-Lite. Instantiated only when DEBUG_CAPTURE != 0.
//
//   Why this exists: ws_quant_unit's q8_out/q8_valid (and the raw
//   sum_out/sum_valid under them) are a LIVE, column-skewed streaming
//   tap with no storage anywhere in the design. Without this module
//   there is nothing for a peek to read - results only exist for the
//   cycle they stream past. This is a bring-up aid, not the data
//   path; DMA/AXI-HP remains the real result path.
//
//   Deskew: column j's values arrive on their own schedule (column j
//   lags column 0 by j cycles), so each column owns its own write
//   index. This is exactly ws_tiling_engine's accumulate-on-write
//   bank shape, minus the accumulate - one bank per column, written
//   in arrival order. Per-column write ports are what keep these in
//   fabric/LUTRAM rather than BRAM.
//
//   Arming: v_cnt[j] is cleared by `arm` (the core go pulse), so
//   every run rewrites its banks from vector 0 and a second run
//   cannot leave stale values at any index it writes. Indices beyond
//   a run's n_vecs deliberately retain the previous run's data -
//   hosts read 0..n_vecs-1.
//
//   Both streams are captured: the INT8 q8 values AND the raw INT32
//   accumulator sums, so a hardware mismatch can be localised to the
//   array (sums wrong) or the requant (sums right, q8 wrong) without
//   another build.
//
//   Read port is REGISTERED (rd_data valid one cycle after rd_en) so
//   the wide element mux never sits on a combinational path - it must
//   not lengthen the core-domain critical path, which is already
//   route-bound. Zero DSPs: storage and muxing only.
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

    // bottom-edge streams (column-skewed, one valid per column)
    input  logic signed [ACC_WIDTH-1:0] sum_in    [N],
    input  logic signed [OUT_WIDTH-1:0] q8_in     [N],
    input  logic                        val_in    [N],

    // registered debug read: rd_data valid one cycle after rd_en
    input  logic                        rd_en,
    input  logic                        rd_is_sum,  // 0 = q8 lane, 1 = INT32
    input  logic [VEC_W-1:0]            rd_vec,
    input  logic [3:0]                  rd_idx,     // q8: lane; sum: column
    output logic [31:0]                 rd_data
);

    // The q8 read packs four INT8 values into one 32-bit element.
    if (OUT_WIDTH != 8) begin : gen_out_width_guard
        initial begin
            $error("ws_result_capture: q8 lane packing assumes OUT_WIDTH == 8 (got %0d)",
                   OUT_WIDTH);
            $fatal(1);
        end
    end

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
    // Banks. Stored as RAW BITS (unsigned) - readback is byte-exact
    // and the host applies the interpretation, which also sidesteps
    // the unreliable signedness of unpacked-array elements here.
    // Banks are datapath (no reset); v_cnt is control (async reset).
    //----------------------------------------------------------------
    logic [OUT_WIDTH-1:0] q8_bank  [N][MAX_N_VECS];
    logic [ACC_WIDTH-1:0] sum_bank [N][MAX_N_VECS];
    logic [VEC_W-1:0]     v_cnt    [N];

    for (genvar j = 0; j < N; j++) begin : gen_cap
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                v_cnt[j] <= '0;
            else if (arm)
                v_cnt[j] <= '0;
            else if (val_in[j])
                v_cnt[j] <= v_cnt[j] + 1'b1;
        end

        always_ff @(posedge clk) begin
            if (val_in[j]) begin
                q8_bank[j][v_cnt[j]]  <= q8_in[j];
                sum_bank[j][v_cnt[j]] <= sum_in[j];
            end
        end
    end

    //----------------------------------------------------------------
    // Registered read. The q8 lane gathers columns lane*4 .. lane*4+3
    // (element 0 in bits [7:0], matching the buffer word convention).
    // Both gathers are BOUNDS-CHECKED against N: this module is built
    // at N=4 as well as N=16 (tb_axi_lite_ctrl builds ws_tpu_top at
    // N=4), where only lane 0 exists and columns 4..15 do not. Out of
    // range reads return 0 rather than indexing off the end.
    //----------------------------------------------------------------
    logic [31:0] q8_word;
    logic [31:0] sum_word;
    int          q8_idx;

    always_comb begin
        q8_word = 32'd0;
        for (int k = 0; k < 4; k++) begin
            q8_idx = rd_idx[1:0] * 4 + k;
            if (q8_idx < N)
                q8_word[k*8 +: 8] = q8_bank[q8_idx][rd_vec];
        end
    end

    always_comb begin
        sum_word = 32'd0;
        if (rd_idx < N)
            sum_word = sum_bank[rd_idx][rd_vec];
    end

    always_ff @(posedge clk) begin
        if (rd_en) begin
            if (rd_is_sum) rd_data <= sum_word;
            else           rd_data <= q8_word;
        end
    end

endmodule
