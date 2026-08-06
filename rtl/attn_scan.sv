`timescale 1ns / 1ps
//====================================================================
// attn_scan: streaming attention over the quantized KV cache, one
// Q head. Online (FlashAttention-style) softmax, single pass.
//
//   Single pass is not a preference, it is the budget. The read side
//   is DRAM-bound - 991 MB/token at ctx 40960 - so a two-pass softmax
//   (find max, then weight) would re-read the whole cache and DOUBLE
//   token latency. Materializing the score row is impossible anyway at
//   unbounded context.
//
//   Step 2 of the read spec: ONE Q head, no GQA reuse yet.
//
//====================================================================
// THE RESCALE IS A SHIFT, NOT AN exp()
//   When a new position's score beats the running max, every
//   previously accumulated quantity must be scaled by 2^(m_old-m_new)
//   - including the 128-wide output accumulator, not just the scalar
//   sum. That is the expensive mechanic, so it is worth removing
//   rather than optimizing.
//
//   Work in base 2 (log2(e) folds into the query scale beside
//   1/sqrt(d_head), free) and keep the running max INTEGRAL. Then the
//   rescale exponent is an integer and the factor is a power of two:
//
//       o[i] >>>= delta        l >>= delta
//
//   No LUT, no multiplier, and ZERO multiplicative error - a power of
//   two is exact in fixed point. The residual cost is truncation of
//   low bits, and that is COMMON-MODE: o and l take the same exact
//   factor and the answer is o/l, so the factor cancels identically.
//   GUARD fractional bits below the table's Q1.15 exist to absorb that
//   truncation.
//
//   Price: the running max sits up to 1 unit above the true max, so
//   the largest weight lands in [0.5, 1] rather than exactly 1. One
//   bit of dynamic range.
//
//   Rescales fire ~ln(L) times for exchangeable scores (about 9 at
//   L=8192), but recency bias makes monotonically rising scores
//   plausible, so the datapath absorbs the worst case rather than
//   assuming the average.
//
//====================================================================
// K IS NOW INT8, SO THE K INNER LOOP HAS A REAL MULTIPLIER
//   The earlier "no multiplier in either inner loop" result depended
//   on K being 2-bit: a 4:1 mux over products precomputed once per
//   head. The end-to-end study moved K to INT8 (top-1 53.3% -> 91.2%),
//   and at 8 bits there are 256 values per coordinate, so the K side
//   is a genuine signed MAC - about LANES DSPs. V stays 3-bit and
//   stays mux-based, so the V loop is still multiplier-free.
//
//   K is also no longer rotated, which removes work rather than adding
//   it: no query rotation, no pk precompute, no per-vector norm, no
//   QJL term. The score is simply sum_i yq[i]*k[i].
//
// (historical, for the V path)
//   Position-dependent scales factor OUT of the coordinate sum, and
//   what remains is position-INDEPENDENT, so both dot products become
//   selection networks over precomputed products:
//
//     score = (vnorm/2^15)*SUM_i pk[i][codeK[i]] + amp*SUM_i (+-yq[i])
//                                ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^
//                                4:1 mux + adder   add/sub by sign bit
//     o[i] += pv[codeV[i]]       ^ 8 products, precomputed per position
//
//   pk[i][k] = yq[i]*levK[k] is built ONCE per head in S_PREP, because
//   levK is a constant codebook - it does not depend on the position.
//   So the whole scan is muxes and adders; multiplies happen ~11 times
//   per POSITION, never per coordinate.
//
//   If synthesis reports DSPs scaling with LANES, that factoring did
//   not happen and something reintroduced a per-coordinate multiply.
//
//====================================================================
// LANES = 1 IN THIS STEP, AND THAT IS A KNOWN THROUGHPUT GAP
//   The DRAM budget is 15.5 cycles per cache entry, i.e. ~16
//   coordinates/cycle. This module does one, so it is ~16x short. The
//   fix is pure replication - LANES copies of the mux+adder plus
//   banking pk by lane - which is mechanical and changes no arithmetic.
//   Step 2 is about the softmax mechanic being right; widening is a
//   throughput step, deliberately not conflated with it.
//
//====================================================================
// WHAT IT CONSUMES
//   kv_unpack, twice - K and V have different formats (K is 2-bit MSE
//   plus a QJL residual, V is pure 3-bit MSE). It does NOT consume
//   kv_dequant: that reconstructs into the ORIGINAL domain, which
//   would cost a 448-cycle inverse butterfly per entry, 1024x the
//   budget. Scores are inner products and the rotation is orthogonal,
//   so K's rotation cancels; V's is linear, so it commutes with the
//   weighted sum and comes out ONCE at the end of the scan.
//
//   Hence the output here is the accumulator o[] and the normalizer l,
//   still in the ROTATED domain and un-normalized. Dividing and
//   inverse-rotating are one-per-head operations that belong outside
//   the scan loop.
//====================================================================
module attn_scan #(
    parameter int N       = 128,
    parameter int QBITS_K = 8,   // K is plain INT8 now
    parameter int QBITS_V = 3,
    parameter int SSHIFT  = 12,   // score -> base-2 logit, Q16.16
    parameter int SFRAC   = 16,
    parameter int GUARD   = 16,   // fractional bits below the table's Q1.15
    parameter int MAXN    = 40,   // weight underflow cutoff, in octaves
    parameter CBV_PATH    = "rtl/kv_cb_m0_b3.memh",
    parameter EXP2_PATH   = "rtl/exp2_lut.memh"
)(
    input  logic clk,
    input  logic rst_n,

    // ---- per-head setup: rotated, pre-scaled query ----
    input  logic                 q_wr,
    input  logic [$clog2(N)-1:0] q_addr,
    input  logic signed [31:0]   q_data,
    input  logic                 prep_start,
    output logic                 prep_done,

    // ---- scan control ----
    input  logic                 scan_start,   // clears m, l, o
    input  logic                 pos_last,     // this position ends the scan
    output logic                 busy,
    output logic                 pos_done,
    output logic                 scan_done,

    // ---- cache word streams ----
    input  logic                 k_w_valid,
    input  logic [31:0]          k_w_data,
    input  logic                 v_w_valid,
    input  logic [31:0]          v_w_data,

    // ---- result, rotated domain, un-normalized ----
    input  logic [$clog2(N)-1:0] o_rd_addr,
    output logic signed [47:0]   o_rd_data,
    output logic [47:0]          l_out,
    output logic signed [15:0]   m_out
);

    localparam int AW     = $clog2(N);
    localparam int NLEVV  = 1 << QBITS_V;
    localparam int PKW    = 40;
    localparam int OW     = 48;
    localparam int LW     = 48;
    localparam int WW     = 32;

    localparam logic [AW:0] N_LAST = N - 1;

    (* rom_style = "distributed" *) logic [15:0] cbv [2*NLEVV-1];
    (* rom_style = "block"       *) logic [15:0] e2  [257];
    initial begin
        $readmemh(CBV_PATH,  cbv);
        $readmemh(EXP2_PATH, e2);
    end

    //----------------------------------------------------------------
    // Decoders. Two, because K and V are different formats.
    //----------------------------------------------------------------
    logic [N*QBITS_K-1:0] k_codes;
    logic [N-1:0]         k_signs;
    logic [29:0]          k_vnorm, k_rnorm;
    logic                 k_entry, k_busy;

    logic [N*QBITS_V-1:0] v_codes;
    logic [N-1:0]         v_signs_nc;
    logic [29:0]          v_vnorm, v_rnorm_nc;
    logic                 v_entry, v_busy;

    kv_unpack #(.N(N), .QBITS(QBITS_K), .QJL_EN(0), .HAS_NORM(0)) u_k (
        .clk(clk), .rst_n(rst_n), .w_valid(k_w_valid), .w_data(k_w_data),
        .busy(k_busy), .codes(k_codes), .signs(k_signs),
        .vnorm(k_vnorm), .rnorm(k_rnorm), .entry_valid(k_entry));

    kv_unpack #(.N(N), .QBITS(QBITS_V), .QJL_EN(0)) u_v (
        .clk(clk), .rst_n(rst_n), .w_valid(v_w_valid), .w_data(v_w_data),
        .busy(v_busy), .codes(v_codes), .signs(v_signs_nc),
        .vnorm(v_vnorm), .rnorm(v_rnorm_nc), .entry_valid(v_entry));

    // V words can arrive before the K phase finishes, so latch.
    logic v_pend;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)             v_pend <= 1'b0;
        else if (v_entry)       v_pend <= 1'b1;
        else if (pos_done)      v_pend <= 1'b0;
    end

    //----------------------------------------------------------------
    // Storage
    //----------------------------------------------------------------
    logic signed [31:0]    yq [N];
    logic signed [OW-1:0]  o  [N];

    logic [AW:0]           cnt;
    logic signed [15:0]    m_run;
    logic [LW-1:0]         l_run;
    logic                  last_q;

    typedef enum logic [3:0] {
        S_IDLE, S_PREP, S_KDOT, S_SCORE, S_WEIGHT, S_RESC, S_WAITV,
        S_VACC, S_PDONE, S_SDONE
    } state_t;
    state_t state;

    assign busy      = (state != S_IDLE);
    assign o_rd_data = o[o_rd_addr];
    assign l_out     = l_run;
    assign m_out     = m_run;

    always_ff @(posedge clk) begin
        if (q_wr) yq[q_addr] <= q_data;
    end

    //----------------------------------------------------------------
    // S_PREP: pk[i][k] = yq[i] * levK[k]. Four multiplies per cycle,
    // N cycles, once per head. This is what makes the scan's inner
    // loop multiplier-free.
    //----------------------------------------------------------------
    //----------------------------------------------------------------
    // S_KDOT: the selection network. One 4:1 mux and two adders.
    //----------------------------------------------------------------
    logic [AW-1:0]      ci;
    logic [7:0]         kraw;
    logic signed [8:0]  k_s;
    logic signed [31:0] yq_i;
    logic signed [47:0] kprod;
    logic signed [63:0] dotA;

    assign ci    = cnt[AW-1:0];
    assign kraw  = k_codes[(N-1-ci)*8 +: 8];
    assign k_s   = {kraw[7], kraw};
    assign yq_i  = yq[ci];
    assign kprod = yq_i * k_s;          // the one real MAC per lane

    always_ff @(posedge clk) begin
        if (state == S_SCORE || state == S_IDLE) begin
            dotA <= '0;
        end else if (state == S_KDOT) begin
            dotA <= dotA + {{16{kprod[47]}}, kprod};
        end
    end

    //----------------------------------------------------------------
    // S_SCORE: fold the per-entry scalars back in. Two multiplies.
    //----------------------------------------------------------------
    logic signed [79:0] score;
    logic signed [31:0] sprime, sprime_q;

    // Arithmetic shift of the full-width signed value, then truncate.
    // A part-select would drop the sign bit and turn negative logits
    // positive, and attention logits are routinely negative.
    // No per-vector norm and no QJL residual on an INT8 K line, so the
    // score is the raw dot product; all constant scaling (both cache
    // scales, 1/sqrt(d_head), log2(e)) is folded into yq once per head.
    assign score  = {{16{dotA[63]}}, dotA};
    assign sprime = (score >>> SSHIFT);

    always_ff @(posedge clk) begin
        if (state == S_SCORE) sprime_q <= sprime;
    end

    //----------------------------------------------------------------
    // S_WEIGHT: integer max update, then 2^-n * 2^-f.
    //----------------------------------------------------------------
    logic               first;
    logic signed [15:0] m_cand, m_next;
    logic [15:0]        resc_d;
    logic signed [47:0] mshift, dpos;
    logic [31:0]        nsh;
    logic [7:0]         fidx, ffrac;
    logic [15:0]        ea, eb, eval15;
    logic signed [17:0] ediff;
    logic signed [31:0] eterm;
    logic [WW-1:0]      wval, wshifted;

    assign m_cand = (sprime_q >>> SFRAC) + 16'sd1;
    // The first position has no previous max to rescale against, so it
    // adopts m_cand outright. Without this the initial sentinel would
    // produce an enormous resc_d whose low bits are all that survive
    // the shift - harmless while the accumulators are zero, but only
    // by accident.
    assign m_next = first ? m_cand : ((m_cand > m_run) ? m_cand : m_run);
    assign resc_d = first ? 16'd0  : (m_next - m_run);

    assign mshift = {{(48-16){m_next[15]}}, m_next} <<< SFRAC;
    assign dpos   = mshift - {{16{sprime_q[31]}}, sprime_q};  // >= 0 by construction
    assign nsh    = dpos[47:SFRAC];
    assign fidx   = dpos[SFRAC-1 -: 8];
    assign ffrac  = dpos[SFRAC-9 -: 8];

    // Exactly the model in tb/gen_attn_exp2.py:
    //   got = tab[i] + (((tab[i+1]-tab[i]) * frac) >> 8)
    // Verilog >>> on a signed value floors, as Python's >> does, so
    // the negative difference truncates identically in both.
    assign ea     = e2[fidx];
    assign eb     = e2[fidx + 1];
    assign ediff  = $signed({2'b0, eb}) - $signed({2'b0, ea});
    assign eterm  = ($signed({{14{ediff[17]}}, ediff}) * $signed({1'b0, ffrac})) >>> 8;
    assign eval15 = ea + eterm[15:0];
    assign wval   = {{(WW-16){1'b0}}, eval15} << GUARD;
    assign wshifted = (nsh >= MAXN) ? '0 : (wval >> nsh[5:0]);

    logic [WW-1:0] w_q;
    always_ff @(posedge clk) begin
        if (state == S_WEIGHT) w_q <= wshifted;
    end

    //----------------------------------------------------------------
    // S_VACC: eight products per position, then an 8:1 mux per coord.
    //----------------------------------------------------------------
    logic [63:0]          cw;
    logic signed [63:0]   cw_s;
    logic [QBITS_V-1:0]   vcode;
    logic signed [15:0]   lv_sel;
    logic signed [63:0]   pv, pv_sh;
    logic signed [OW-1:0] pv_s;
    logic [4:0]           lvi;

    // w_q and v_vnorm are both non-negative, so cw is unsigned; the
    // level it then multiplies is signed, so pv must be, and its
    // rescale is an arithmetic shift of the full width before
    // truncation - same trap as the score path above.
    assign cw     = w_q * {34'b0, v_vnorm};
    assign cw_s   = $signed({1'b0, cw[62:15]});
    assign vcode  = v_codes[(N-1-ci)*QBITS_V +: QBITS_V];
    assign lvi    = NLEVV[4:0] - 5'd1 + {2'b0, vcode};
    assign lv_sel = cbv[lvi];
    assign pv     = cw_s * lv_sel;
    assign pv_sh  = pv >>> 15;
    assign pv_s   = pv_sh[OW-1:0];

    //----------------------------------------------------------------
    // Accumulators. The rescale is the only place o and l lose bits,
    // and it loses them identically, so o/l is unaffected.
    //----------------------------------------------------------------
    logic signed [OW-1:0] o_cur;
    assign o_cur = o[ci];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            first <= 1'b1;
            m_run <= '0;
            l_run <= '0;
        end else if (prep_start || scan_start) begin
            first <= 1'b1;
            m_run <= '0;
            l_run <= '0;
        end else if (state == S_WEIGHT) begin
            first <= 1'b0;
            m_run <= m_next;
            l_run <= (l_run >> resc_d[5:0]) + {{(LW-WW){1'b0}}, wshifted};
        end
    end

    // resc_d is combinational from (m_next - m_run), and m_run is
    // updated at the END of S_WEIGHT - so by the time S_RESC runs it
    // has already collapsed to zero and the shift would be a no-op.
    // l escapes this because its rescale happens inside S_WEIGHT while
    // the value is still live; o[], which needs its own N-cycle pass,
    // does not. Latch it.
    logic [15:0] resc_q;
    always_ff @(posedge clk) begin
        if (state == S_WEIGHT) resc_q <= resc_d;
    end

    // S_PREP runs exactly N cycles, which is also what zeroing o[]
    // needs, so the two share the pass rather than costing a state.
    always_ff @(posedge clk) begin
        if (state == S_PREP) begin
            o[ci] <= '0;
        end else if (state == S_RESC) begin
            o[ci] <= o_cur >>> resc_q[5:0];
        end else if (state == S_VACC) begin
            o[ci] <= o_cur + pv_s;
        end
    end

    //----------------------------------------------------------------
    // Sequencer
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            cnt       <= '0;
            prep_done <= 1'b0;
            pos_done  <= 1'b0;
            scan_done <= 1'b0;
            last_q    <= 1'b0;
        end else begin
            prep_done <= 1'b0;
            pos_done  <= 1'b0;
            scan_done <= 1'b0;
            case (state)
                S_IDLE: begin
                    cnt <= '0;
                    if (prep_start)     state <= S_PREP;
                    else if (k_entry)   state <= S_KDOT;
                end
                S_PREP: if (cnt == N_LAST) begin
                    state     <= S_IDLE;
                    cnt       <= '0;
                    prep_done <= 1'b1;
                end else cnt <= cnt + 1'b1;
                S_KDOT: if (cnt == N_LAST) begin
                    state <= S_SCORE;
                    cnt   <= '0;
                end else cnt <= cnt + 1'b1;
                S_SCORE:  state <= S_WEIGHT;
                S_WEIGHT: begin
                    last_q <= pos_last;
                    if (resc_d != 16'd0) state <= S_RESC;
                    else                 state <= S_WAITV;
                    cnt <= '0;
                end
                S_RESC: if (cnt == N_LAST) begin
                    state <= S_WAITV;
                    cnt   <= '0;
                end else cnt <= cnt + 1'b1;
                S_WAITV: if (v_pend) begin
                    state <= S_VACC;
                    cnt   <= '0;
                end
                S_VACC: if (cnt == N_LAST) begin
                    state    <= S_PDONE;
                    cnt      <= '0;
                    pos_done <= 1'b1;
                end else cnt <= cnt + 1'b1;
                S_PDONE: begin
                    if (last_q) begin
                        state     <= S_SDONE;
                        scan_done <= 1'b1;
                    end else begin
                        state <= S_IDLE;
                    end
                end
                S_SDONE: state <= S_IDLE;
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
