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
//   INT32 ONLY - the INT8 stream is deliberately NOT captured.
//   Measured on the routed build: the combinational chain
//   array sum_reg -> requant (32x16 multiply + round/shift + int8
//   saturate) -> a capturing flop is 8.355ns, of which 4.798-5.037ns
//   is LOGIC alone, against a 4.700ns period. That block cannot close
//   at this clock no matter where registers are placed AROUND it, and
//   it cannot be multicycled because q8_valid is asserted on
//   back-to-back cycles while vectors stream. Capturing q8 would
//   require pipelining inside requant, which is shared by three other
//   call sites. Hosts apply the requant math to the INT32 values
//   instead - which is what a PYNQ driver does anyway, and requant
//   itself is covered exhaustively by tb_requant.
//   NOTE for the future DMA path and for ws_tpu_tiled_top: any
//   SYNCHRONOUS consumer of a requant output hits this same wall.
//   ws_tiling_engine already registers rq_final into res_q8, so the
//   tiled top carries the same latent violation - it has simply never
//   been synthesized.
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

    // registered debug read: rd_data valid one cycle after rd_en
    input  logic                        rd_en,
    input  logic [VEC_W-1:0]            rd_vec,
    input  logic [3:0]                  rd_idx,     // column
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
    end

    //----------------------------------------------------------------
    // Registered read, bounds-checked against N: this module is built
    // at N=4 as well as N=16 (tb_axi_lite_ctrl builds ws_tpu_top at
    // N=4), where columns 4..15 do not exist.
    //----------------------------------------------------------------
    logic [31:0] sum_word;

    always_comb begin
        sum_word = 32'd0;
        if (rd_idx < N)
            sum_word = sum_bank[rd_idx][rd_vec];
    end

    always_ff @(posedge clk) begin
        if (rd_en)
            rd_data <= sum_word;
    end

endmodule
