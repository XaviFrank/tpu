`timescale 1ns / 1ps
//====================================================================
// attn_kv_top: the whole attention unit as one synthesizable block -
// KV quantized write path, on-chip cache, and the GQA-shared streaming
// read path, behind a single repeatable go/busy/done.
//
//   xk[] -> kv_pack_int8 --------------------> kv_cache_store (32 w)
//   xv[] -> kv_rotate -> kv_quantize --------> kv_cache_store (13 w)
//                                                    |
//                              read back per position v
//   q[NQ][] -------------------------------->  attn_scan_gqa -> o[], l
//
//   tb_attn_e2e already assembled these modules and proved the result
//   bit-exact against numpy. What it could not provide is a THING TO
//   SYNTHESIZE: it is a testbench, so the sequencing that drives the
//   assembly lives in an initial block. This module is that sequencing
//   in RTL, and nothing else - the datapath is the same verified
//   instances, unmodified.
//
//====================================================================
// ONE go = ONE DECODE STEP, FOR ONE (LAYER, KV-HEAD)
//   Append this token's K and V, then score the query against every
//   position in the cache including the one just appended. That is
//   exactly the per-token attention operation, so:
//
//     - causal masking is structural, not a mask. The cache holds only
//       positions <= t because only those have been appended.
//     - repeated invocation is the normal mode, not a stress case. The
//       t-th go scans t+1 positions.
//     - the accumulators are reset by prep_start every invocation, so
//       nothing leaks between calls. o[] is zeroed in S_PREP, and
//       (m, l, first) are cleared by the same pulse.
//
//   NOT bundled: the queries. q_wr writes attn_scan's yq[] directly and
//   those registers survive prep_start, so a caller may load a fresh
//   query per token (the real case) or load once and invoke repeatedly.
//   Both are exercised by tb_attn_kv_top.
//
//====================================================================
// ONE CACHE, NUM_LAYERS REGIONS - `layer` selects which
//   A transformer layer's KV cache is private to that layer, so a
//   28-layer decode needs 28 of them. This block does NOT own 28
//   stores; it owns one K store and one V store carved into NUM_LAYERS
//   regions, and `layer` picks the region for the whole invocation.
//
//   Instantiating 28 stores would have cost the same on-chip memory
//   and neither approach reaches useful context (~1.65 GB at 40960),
//   so the tie-breaker was which one survives the move to DRAM. A
//   layer index folded into an address is exactly what a DRAM-backed
//   cache needs; 28 separate instances would have to be collapsed back
//   into one address space to talk to a memory controller at all. Full
//   reasoning in kv_cache_store's header.
//
//   `layer` is LATCHED at go. Both write and read phases of one
//   invocation then use the sampled value, so a caller that changes
//   `layer` mid-invocation cannot write layer A's line and read layer
//   B's cache. While idle the outputs follow the live input, so
//   `n_pos` can be polled per layer without starting anything.
//
//====================================================================
// WHY THE SEQUENCER WAITS FOR !busy BETWEEN POSITIONS
//   The NQ attn_scan instances see one broadcast word stream but do NOT
//   run in lockstep - a head whose running max moves spends an extra
//   N-cycle rescale pass and a head whose does not, does not. So
//   pos_done (which is the AND across heads) can assert while a
//   trailing head is still in S_PDONE. Feeding the next position's
//   words then would push words into a decoder that is still busy.
//   T_POSRDY waits for the gqa-level busy to clear, which is what
//   tb_attn_e2e's `while (s_busy)` was doing by hand.
//====================================================================
module attn_kv_top #(
    parameter int N        = 128,   // head_dim
    parameter int NQ       = 2,     // Q heads sharing this KV head
    parameter int MAX_POS  = 102,   // cache capacity, in positions PER LAYER
    parameter int NUM_LAYERS = 1,   // transformer layers sharing the cache
    parameter int QBITS_V  = 3,
    parameter int ROT_MODE = 0,     // V-side rotation: 0 = Hadamard
    parameter int SSHIFT   = 0,
    // $readmemh resolves against the CWD, and Vivado's -generic only
    // reaches ROOT-module parameters - so every table path used
    // anywhere below is a parameter HERE and forwarded down. Leaving
    // one on a child silently initialises that ROM to zero, which is
    // the rmsnorm/swiglu_gate/rope failure (commit e5489dc).
    //   -generic CB_PATH=C:/dev/tpu/rtl/kv_cb_m0_b3.memh
    //   -generic EXP2_PATH=C:/dev/tpu/rtl/exp2_lut.memh
    //   -generic ROT_SIGN_PATH=C:/dev/tpu/rtl/kv_rot_sign.memh
    //   -generic ROT_PI_PATH=C:/dev/tpu/rtl/kv_rot_pi.memh
    //
    // ONE codebook parameter, not two. kv_quantize's thresholds and
    // attn_scan's reconstruction levels must come from the SAME table
    // or the write and read sides disagree about what a code means.
    // Separate parameters would make that misconfigurable.
    parameter CB_PATH       = "rtl/kv_cb_m0_b3.memh",
    parameter EXP2_PATH     = "rtl/exp2_lut.memh",
    parameter ROT_SIGN_PATH = "rtl/kv_rot_sign.memh",
    parameter ROT_PI_PATH   = "rtl/kv_rot_pi.memh",
    // DERIVED - do not override. Same reason as kv_cache_store's LAYW:
    // a port width cannot reference a localparam, and $clog2(1) is 0.
    parameter int LAYW      = (NUM_LAYERS <= 1) ? 1
                                                : $clog2(NUM_LAYERS)
)(
    input  logic clk,
    input  logic rst_n,

    // Empties both caches. Ignored while busy - the address arithmetic
    // is live mid-invocation and resetting it there would corrupt the
    // line being written. Apply between invocations.
    input  logic clear,

    // Which layer's cache this invocation appends to and scans.
    // Sampled at go; free to change between invocations.
    input  logic [LAYW-1:0]       layer,

    // ---- this token's K and V, INT8, before quantization ----
    input  logic                  xk_wr,
    input  logic [$clog2(N)-1:0]  xk_addr,
    input  logic signed [7:0]     xk_data,
    input  logic                  xv_wr,
    input  logic [$clog2(N)-1:0]  xv_addr,
    input  logic signed [7:0]     xv_data,

    // ---- this token's queries: rotated, pre-scaled, one per Q head --
    input  logic                  q_wr,
    input  logic [$clog2(NQ)-1:0] q_head,
    input  logic [$clog2(N)-1:0]  q_addr,
    input  logic signed [31:0]    q_data,

    // ---- control ----
    input  logic                  go,
    output logic                  busy,
    output logic                  done,

    // ---- cache status ----
    output logic [$clog2(MAX_POS+1)-1:0] n_pos,
    output logic                  full,
    output logic                  overflow,

    // ---- result: rotated domain, un-normalized, per Q head ----
    input  logic [$clog2(NQ)-1:0] o_rd_head,
    input  logic [$clog2(N)-1:0]  o_rd_addr,
    output logic signed [47:0]    o_rd_data,
    output logic [47:0]           l_out,
    output logic signed [15:0]    m_out
);

    localparam int AW   = $clog2(N);
    localparam int POSW = $clog2(MAX_POS+1);
    localparam int NWK  = (N * 8) / 32;                 // 32
    localparam int NWV  = ((N * QBITS_V) / 32) + 1;     // 13

    localparam logic [AW:0] N_LAST  = N - 1;
    localparam logic [5:0]  NWK_LAST = NWK - 1;
    localparam logic [5:0]  NWV_LAST = NWV - 1;

    //----------------------------------------------------------------
    // Input vector buffers. kv_pack_int8 wants K streamed one
    // coordinate per cycle in order; kv_rotate wants RANDOM access to V
    // with one cycle of read latency (the butterfly reads two arbitrary
    // indices). Hence a plain buffer each rather than a shared FIFO.
    //----------------------------------------------------------------
    logic [7:0] kbuf [N];
    logic [7:0] vbuf [N];

    always_ff @(posedge clk) begin
        if (xk_wr) kbuf[xk_addr] <= xk_data;
    end
    always_ff @(posedge clk) begin
        if (xv_wr) vbuf[xv_addr] <= xv_data;
    end

    //----------------------------------------------------------------
    // Sequencer state, declared first: the write/read datapath below
    // is driven straight off it.
    //----------------------------------------------------------------
    typedef enum logic [3:0] {
        T_IDLE, T_VSTART, T_WFEED, T_WWAIT,
        T_PREP, T_PWAIT, T_POSRDY, T_KRD, T_VRD, T_PDWAIT,
        T_FIN,  T_DONE
    } tstate_t;
    tstate_t tst;

    logic [AW:0]    fcnt;      // K feed counter, 0..N-1
    logic [5:0]     wcnt;      // cache word counter, 0..31
    logic [POSW-1:0] pcnt;     // position under scan
    logic [POSW-1:0] npos_lat; // positions to scan this invocation

    assign busy = (tst != T_IDLE);

    //----------------------------------------------------------------
    // Write path: K
    //----------------------------------------------------------------
    logic               kin_v, kp_busy, kp_wv, kp_done;
    logic signed [23:0] kin_d;
    logic [4:0]         kp_wi;
    logic [31:0]        kp_wd;

    logic [7:0] kfeed;
    assign kfeed = kbuf[fcnt[AW-1:0]];
    assign kin_v = (tst == T_WFEED);
    assign kin_d = {{16{kfeed[7]}}, kfeed};

    kv_pack_int8 #(.N(N)) u_kpack (
        .clk(clk), .rst_n(rst_n), .in_valid(kin_v), .in_data(kin_d),
        .busy(kp_busy), .w_valid(kp_wv), .w_idx(kp_wi), .w_data(kp_wd),
        .done(kp_done));

    //----------------------------------------------------------------
    // Write path: V
    //----------------------------------------------------------------
    logic               vstart, vr_busy, vr_en, vr_done;
    logic [AW-1:0]      vr_ad;
    logic signed [7:0]  vr_dt;
    logic               vy_v;
    logic signed [23:0] vy_d;
    logic               vq_busy, vq_wv, vq_done;
    logic [4:0]         vq_wi;
    logic [31:0]        vq_wd;

    assign vstart = (tst == T_VSTART);

    always_ff @(posedge clk) vr_dt <= vbuf[vr_ad];

    kv_rotate #(.N(N), .ROT_MODE(ROT_MODE),
                .ROT_SIGN_PATH(ROT_SIGN_PATH),
                .ROT_PI_PATH(ROT_PI_PATH)) u_vrot (
        .clk(clk), .rst_n(rst_n), .start(vstart), .busy(vr_busy),
        .x_rd_en(vr_en), .x_rd_addr(vr_ad), .x_rd_data(vr_dt),
        .y_valid(vy_v), .y_idx(), .y_out(vy_d), .done(vr_done));

    kv_quantize #(.N(N), .QBITS(QBITS_V), .QJL_EN(0),
                  .CB_PATH(CB_PATH)) u_vq (
        .clk(clk), .rst_n(rst_n), .in_valid(vy_v), .in_data(vy_d),
        .busy(vq_busy), .w_valid(vq_wv), .w_idx(vq_wi), .w_data(vq_wd),
        .done(vq_done));

    //----------------------------------------------------------------
    // The caches. clear is gated on !busy - see the port comment.
    //----------------------------------------------------------------
    logic        st_clear;
    logic [POSW-1:0] kn_pos, vn_pos;
    logic        k_full, k_ovf, v_full, v_ovf;
    logic        krd_en, vrd_en;
    logic [POSW-1:0] krd_pos, vrd_pos;
    logic [4:0]  krd_wi, vrd_wi;
    logic [31:0] krd_d, vrd_d;

    assign st_clear = clear && !busy;

    // Latched at go, live while idle - see the header. Both stores and
    // both ports see the same value, because one invocation is one
    // layer by construction.
    logic [LAYW-1:0] layer_lat, layer_eff;
    assign layer_eff = busy ? layer_lat : layer;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)                     layer_lat <= '0;
        else if ((tst == T_IDLE) && go) layer_lat <= layer;
    end

    kv_cache_store #(.WORDS_PER_VEC(NWK), .MAX_POS(MAX_POS),
                     .NUM_LAYERS(NUM_LAYERS)) u_kstore (
        .clk(clk), .rst_n(rst_n), .clear(st_clear), .w_layer(layer_eff),
        .w_valid(kp_wv), .w_idx(kp_wi), .w_data(kp_wd), .commit(kp_done),
        .n_pos(kn_pos), .full(k_full), .overflow(k_ovf),
        .rd_en(krd_en), .rd_layer(layer_eff),
        .rd_pos(krd_pos), .rd_widx(krd_wi), .rd_data(krd_d));

    kv_cache_store #(.WORDS_PER_VEC(NWV), .MAX_POS(MAX_POS),
                     .NUM_LAYERS(NUM_LAYERS)) u_vstore (
        .clk(clk), .rst_n(rst_n), .clear(st_clear), .w_layer(layer_eff),
        .w_valid(vq_wv), .w_idx(vq_wi), .w_data(vq_wd), .commit(vq_done),
        .n_pos(vn_pos), .full(v_full), .overflow(v_ovf),
        .rd_en(vrd_en), .rd_layer(layer_eff),
        .rd_pos(vrd_pos), .rd_widx(vrd_wi), .rd_data(vrd_d));

    // The two stores are written by independent producers, so a
    // disagreement in their position counts is a real symptom (a
    // dropped commit) rather than a redundant check. Fold it into
    // overflow: the caller already treats that as fatal.
    logic pos_mismatch;
    assign pos_mismatch = (kn_pos != vn_pos);

    assign n_pos    = kn_pos;
    assign full     = k_full || v_full;
    assign overflow = k_ovf || v_ovf || pos_mismatch;

    //----------------------------------------------------------------
    // Read path. The store's read port is registered, so the scan's
    // word-valid is the read enable delayed exactly one cycle.
    //----------------------------------------------------------------
    logic prep_start, prep_done, pos_last, s_busy, pos_done;
    logic kfv, vfv;
    logic in_pos;
    logic [POSW-1:0] last_idx;

    assign krd_en = (tst == T_KRD);
    assign vrd_en = (tst == T_VRD);
    assign krd_wi = wcnt[4:0];
    assign vrd_wi = wcnt[4:0];
    assign krd_pos = pcnt;
    assign vrd_pos = pcnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kfv <= 1'b0;
            vfv <= 1'b0;
        end else begin
            kfv <= krd_en;
            vfv <= vrd_en;
        end
    end

    assign prep_start = (tst == T_PREP);
    assign last_idx   = npos_lat - 1'b1;
    assign in_pos     = (tst == T_POSRDY) || (tst == T_KRD)
                     || (tst == T_VRD)    || (tst == T_PDWAIT);
    assign pos_last   = in_pos && (pcnt == last_idx);

    attn_scan_gqa #(.N(N), .NQ(NQ), .SSHIFT(SSHIFT),
                    .CBV_PATH(CB_PATH), .EXP2_PATH(EXP2_PATH)) u_scan (
        .clk(clk), .rst_n(rst_n),
        .q_wr(q_wr), .q_head(q_head), .q_addr(q_addr), .q_data(q_data),
        .prep_start(prep_start), .prep_done(prep_done),
        .pos_last(pos_last), .busy(s_busy), .pos_done(pos_done),
        .k_w_valid(kfv), .k_w_data(krd_d),
        .v_w_valid(vfv), .v_w_data(vrd_d),
        .o_rd_head(o_rd_head), .o_rd_addr(o_rd_addr),
        .o_rd_data(o_rd_data), .l_out(l_out), .m_out(m_out));

    //----------------------------------------------------------------
    // Write-completion tracking. Waiting on "nothing is busy" ALONE
    // would fall straight through on the first cycle of T_WWAIT if any
    // producer had not started yet; requiring both done pulses first
    // makes the wait unconditional on ordering.
    //----------------------------------------------------------------
    logic kp_seen, vq_seen, w_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kp_seen <= 1'b0;
            vq_seen <= 1'b0;
        end else if (tst == T_IDLE) begin
            kp_seen <= 1'b0;
            vq_seen <= 1'b0;
        end else begin
            if (kp_done) kp_seen <= 1'b1;
            if (vq_done) vq_seen <= 1'b1;
        end
    end

    assign w_ready = kp_seen && vq_seen
                  && !kp_busy && !vr_busy && !vq_busy;

    //----------------------------------------------------------------
    // Sequencer
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tst      <= T_IDLE;
            fcnt     <= '0;
            wcnt     <= '0;
            pcnt     <= '0;
            npos_lat <= '0;
            done     <= 1'b0;
        end else begin
            done <= 1'b0;
            case (tst)
                T_IDLE: begin
                    fcnt <= '0;
                    if (go) tst <= T_VSTART;
                end
                // kv_rotate latches start and begins reading vbuf on
                // its own; the K feed runs beside it from the next
                // cycle, so the two write paths overlap completely.
                T_VSTART: begin
                    tst  <= T_WFEED;
                    fcnt <= '0;
                end
                T_WFEED: if (fcnt == N_LAST) begin
                    tst  <= T_WWAIT;
                    fcnt <= '0;
                end else begin
                    fcnt <= fcnt + 1'b1;
                end
                T_WWAIT: if (w_ready) begin
                    tst      <= T_PREP;
                    npos_lat <= kn_pos;
                end
                T_PREP:  tst <= T_PWAIT;
                T_PWAIT: if (prep_done) begin
                    tst  <= T_POSRDY;
                    pcnt <= '0;
                end
                T_POSRDY: if (!s_busy) begin
                    tst  <= T_KRD;
                    wcnt <= '0;
                end
                T_KRD: if (wcnt == NWK_LAST) begin
                    tst  <= T_VRD;
                    wcnt <= '0;
                end else begin
                    wcnt <= wcnt + 1'b1;
                end
                T_VRD: if (wcnt == NWV_LAST) begin
                    tst  <= T_PDWAIT;
                    wcnt <= '0;
                end else begin
                    wcnt <= wcnt + 1'b1;
                end
                T_PDWAIT: if (pos_done) begin
                    if (pcnt == last_idx) begin
                        tst <= T_FIN;
                    end else begin
                        tst  <= T_POSRDY;
                        pcnt <= pcnt + 1'b1;
                    end
                end
                T_FIN: if (!s_busy) tst <= T_DONE;
                T_DONE: begin
                    tst  <= T_IDLE;
                    done <= 1'b1;
                end
                default: tst <= T_IDLE;
            endcase
        end
    end

endmodule
