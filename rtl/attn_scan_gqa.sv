`timescale 1ns / 1ps
//====================================================================
// attn_scan_gqa: one KV-head scan serving all the Q heads that share
// it. Step 3 of the read spec.
//
//====================================================================
// WHAT THE REUSE ACTUALLY BUYS, AND WHAT IT DOES NOT
//   Qwen3-0.6B maps 16 Q heads onto 8 KV heads, 2:1. Section 2 sized
//   the alternative - re-scanning the cache once per Q head - and the
//   cost was never compute:
//
//                        shared      per-Q-head
//     DRAM traffic       991 MB      1982 MB     (ctx 40960)
//     token latency      661 ms      1322 ms
//     dot-product work   2x          2x  (identical)
//
//   Two Q vectors have to meet every K either way. The only question
//   is whether each cache line is FETCHED once or twice, and on a
//   bandwidth-bound read side fetching is the entire cost. So the
//   reuse is a broadcast: ONE word stream, NQ consumers.
//
//   HONEST ABOUT THE SHAPE OF THIS IMPLEMENTATION. Broadcasting to NQ
//   full attn_scan instances also replicates their kv_unpack decoders,
//   which an ideal shared design would fold into one. That costs LUTs
//   and costs NOTHING in DRAM traffic, which is the resource that was
//   actually scarce. Folding the decoders is a real later optimization
//   and deliberately not conflated with getting the reuse correct.
//
//   Each instance keeps its own running (max, sum, output) state, so
//   the parallel softmax states section 2 called for are exactly the
//   per-instance state - no new mechanism, and no way for one head's
//   rescale to touch another's accumulator.
//====================================================================
module attn_scan_gqa #(
    parameter int N      = 128,
    parameter int NQ     = 2,     // Q heads per KV head
    parameter int SSHIFT = 0,
    // Promoted from attn_scan so they are ROOT-module parameters at
    // this top. Vivado's -generic only overrides root parameters, so
    // leaving them on the child means the override silently does
    // nothing and the ROM initialises to zero - which is exactly the
    // failure rmsnorm/swiglu_gate/rope hit (commit e5489dc), where a
    // synth run reported a plausible DSP count from a table that had
    // never loaded. Forwarded to every instance below.
    //   -generic CBV_PATH=C:/dev/tpu/rtl/kv_cb_m0_b3.memh
    //   -generic EXP2_PATH=C:/dev/tpu/rtl/exp2_lut.memh
    parameter CBV_PATH   = "rtl/kv_cb_m0_b3.memh",
    parameter EXP2_PATH  = "rtl/exp2_lut.memh"
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                 q_wr,
    input  logic [$clog2(NQ)-1:0] q_head,
    input  logic [$clog2(N)-1:0] q_addr,
    input  logic signed [31:0]   q_data,
    input  logic                 prep_start,
    output logic                 prep_done,

    input  logic                 pos_last,
    output logic                 busy,
    output logic                 pos_done,

    // ONE cache word stream, broadcast to every sharing Q head
    input  logic                 k_w_valid,
    input  logic [31:0]          k_w_data,
    input  logic                 v_w_valid,
    input  logic [31:0]          v_w_data,

    input  logic [$clog2(NQ)-1:0] o_rd_head,
    input  logic [$clog2(N)-1:0] o_rd_addr,
    output logic signed [47:0]   o_rd_data,
    output logic [47:0]          l_out,
    output logic signed [15:0]   m_out
);

    logic [NQ-1:0]      s_busy, s_pdone, s_pready;
    logic signed [47:0] s_o [NQ];
    logic [47:0]        s_l [NQ];
    logic signed [15:0] s_m [NQ];

    genvar g;
    generate
    for (g = 0; g < NQ; g = g + 1) begin : gen_q
        logic qw;
        assign qw = q_wr && (q_head == g[$clog2(NQ)-1:0]);

        attn_scan #(.N(N), .QBITS_K(8), .QBITS_V(3), .SSHIFT(SSHIFT),
                    .CBV_PATH(CBV_PATH), .EXP2_PATH(EXP2_PATH)) u (
            .clk        (clk),
            .rst_n      (rst_n),
            .q_wr       (qw),
            .q_addr     (q_addr),
            .q_data     (q_data),
            .prep_start (prep_start),
            .prep_done  (s_pready[g]),
            .scan_start (1'b0),
            .pos_last   (pos_last),
            .busy       (s_busy[g]),
            .pos_done   (s_pdone[g]),
            .scan_done  (),
            .k_w_valid  (k_w_valid),
            .k_w_data   (k_w_data),
            .v_w_valid  (v_w_valid),
            .v_w_data   (v_w_data),
            .o_rd_addr  (o_rd_addr),
            .o_rd_data  (s_o[g]),
            .l_out      (s_l[g]),
            .m_out      (s_m[g])
        );
    end
    endgenerate

    // THE INSTANCES DO NOT RUN IN LOCKSTEP. They see the same stream,
    // but a head whose running max moves takes an extra N-cycle rescale
    // pass and a head whose does not, does not - and on real data the
    // two sharing Q heads rescale a different number of times (1 vs 6
    // on the test row). So pos_done has to wait for ALL of them;
    // reporting instance 0 would release the next position while the
    // other head was still accumulating this one.
    logic [NQ-1:0] pd_lat, pd_now;
    assign pd_now = pd_lat | s_pdone;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)        pd_lat <= '0;
        else if (pos_done) pd_lat <= '0;
        else               pd_lat <= pd_now;
    end

    assign busy      = |s_busy;
    assign pos_done  = &pd_now;
    assign prep_done = s_pready[0];
    assign o_rd_data = s_o[o_rd_head];
    assign l_out     = s_l[o_rd_head];
    assign m_out     = s_m[o_rd_head];

endmodule
