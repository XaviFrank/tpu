`timescale 1ns / 1ps
//====================================================================
// controller v5: tile-lifecycle FSM for matmul_unit, with K-walk
// and DOUBLE-BUFFERED staging (prefetch overlap).
//   go (with k_chunks + first tile indices) ->
//     repeat k_chunks times: FEED the active staging bank while a
//     PREFETCH sequencer fills the shadow bank with the next chunk;
//     S_FETCH survives only as the remainder state (waited in only
//     for the prefetch cycles FEED could not hide) ->
//   one SETTLE -> one DRAIN -> IDLE
//
//   Staging: two banks (ping-pong). FEED reads ONLY the active
//   bank; PREFETCH lands ONLY into the shadow bank; the swap is a
//   single active_bank flip at every FEED entry. A simulation-only
//   tripwire errors if a prefetch landing ever targets the active
//   bank (same hazard class as drain/feed overlap).
//
//   PREFETCH sequencer: its own small counter + landing pipe, NOT
//   FSM states. It is the ONLY driver of the tile-buffer read ports
//   - the old S_FETCH issue logic is gone, so the single read port
//   per buffer is shared by construction, never arbitrated. It runs
//   K issue cycles + 1 landing cycle (tile_buffer's registered-read
//   latency) per chunk.
//
//   Chunk handshake: chunk_req is high for exactly ONE cycle per
//   chunk; a_tile_idx/b_tile_idx are registered at the posedge that
//   ENDS the chunk_req cycle. The windows are:
//     chunk 0   : the accepted-go cycle (state==S_IDLE && go)
//     chunk c+1 : the cycle the machine commits to entering chunk
//                 c's FEED (state==S_FETCH && fetch_ready) - i.e.
//                 ON ENTERING FEED, so chunk c+1's prefetch can
//                 start on FEED's first cycle
//   The issuer presents indices combinationally while chunk_req is
//   high (tiling_engine already answers from counters, so it is
//   agnostic to this timing shift). k_chunks (>= 1; 0 is illegal)
//   is latched once, at the accepted go.
//
//   Timing: a prefetch needs K+1 cycles but only K FEED cycles can
//   hide it, so every interior chunk pays exactly 1 remainder cycle
//   in S_FETCH (the landing of the last word); the cold first chunk
//   of a walk pays the full K+1 (nothing to overlap with). The
//   FEED->FETCH loop-back still never touches clear or drain - THE
//   ACCUMULATORS INTENTIONALLY CARRY PARTIAL SUMS ACROSS CHUNKS -
//   and there is no SETTLE between chunks (valid gating covers the
//   gap, per tb_pe_os T2). SETTLE/DRAIN/constants untouched;
//   k_chunks=1 is cycle-identical to v4.
//
//   Perf flags (in_*, chunk_done, tile_done) are explicit Moore
//   outputs for perf_counters - pure observers, they feed no control
//   logic. The raw state enum is never exported. S_INIT (one cycle
//   after reset) is covered by no flag.
//====================================================================
module controller #(
    parameter int N          = 4,
    parameter int K          = 4,
    parameter int DATA_WIDTH = 8,
    parameter int DEPTH      = 64,  // tile-buffer beat-words; DEPTH/K tiles
    parameter int CHUNK_W    = 4    // width of k_chunks / chunk counter
)(
    input  logic clk,
    input  logic rst_n,

    // command interface
    input  logic                          go,
    input  logic [CHUNK_W-1:0]            k_chunks,   // latched with go
    output logic                          busy,

    // chunk handshake (indices sampled while chunk_req is high;
    // also sampled at the accepted go, for chunk 0)
    output logic                          chunk_req,
    input  logic [$clog2(DEPTH/K)-1:0]    a_tile_idx,
    input  logic [$clog2(DEPTH/K)-1:0]    b_tile_idx,

    // tile-buffer read ports (one beat = 4 packed int8s; rd_data
    // valid 1 cycle after rd_en - see tile_buffer)
    output logic                          rd_en_a,
    output logic                          rd_en_b,
    output logic [$clog2(DEPTH)-1:0]      rd_addr_a,
    output logic [$clog2(DEPTH)-1:0]      rd_addr_b,
    input  logic [31:0]                   rd_data_a,
    input  logic [31:0]                   rd_data_b,

    // datapath control (drives matmul_unit's internals)
    output logic                          clear,
    output logic                          drain,
    output logic                          valid_feed,
    output logic signed [DATA_WIDTH-1:0]  a_feed [N],
    output logic signed [DATA_WIDTH-1:0]  b_feed [N],

    // result interface
    output logic                          result_valid,
    output logic [$clog2(N)-1:0]          result_row,

    // perf observation flags (Moore, passive; see header)
    output logic                          in_fetch,
    output logic                          in_feed,
    output logic                          in_settle,
    output logic                          in_drain,
    output logic                          in_idle,
    output logic                          chunk_done,   // 1-cycle strobe
    output logic                          tile_done     // 1-cycle strobe
);

    //----------------------------------------------------------------
    // State encoding
    //----------------------------------------------------------------
    typedef enum logic [2:0] {
        S_INIT,
        S_IDLE,
        S_FETCH,    // remainder wait only; issues nothing itself
        S_FEED,
        S_SETTLE,
        S_DRAIN
    } state_t;

    state_t state, state_next;

    //----------------------------------------------------------------
    // Counters, latched command
    //----------------------------------------------------------------
    localparam int SETTLE_CYCLES = 2*N; // boundary certified, 2N-1 clips the boundary

    logic [$clog2(K)-1:0]             beat_cnt;
    logic [$clog2(SETTLE_CYCLES)-1:0] settle_cnt;
    logic [$clog2(N)-1:0]             drain_cnt;

    // K-walk bookkeeping: which chunk is in flight, out of how many
    logic [CHUNK_W-1:0]               chunk_cnt;
    logic [CHUNK_W-1:0]               k_chunks_q;
    logic                             last_chunk;

    //----------------------------------------------------------------
    // Double-buffered staging: two banks, FEED reads active_bank,
    // PREFETCH lands into pf_bank (always the shadow)
    //----------------------------------------------------------------
    logic                             active_bank;
    logic signed [DATA_WIDTH-1:0]     a_tile [2][N][K];
    logic signed [DATA_WIDTH-1:0]     b_tile [2][K][N];

    //----------------------------------------------------------------
    // PREFETCH sequencer state (counter + landing pipe, not states)
    //----------------------------------------------------------------
    logic                             pf_active;   // issuing/landing a chunk
    logic                             pf_done;     // shadow bank complete
    logic                             pf_bank;     // bank this prefetch lands in
    logic [$clog2(K+1)-1:0]           pf_cnt;      // issue counter, 0..K
    logic                             pf_land_valid;
    logic [$clog2(K)-1:0]             pf_land_cnt; // landing runs 1 cycle behind issue
    logic [$clog2(DEPTH/K)-1:0]       pf_a_idx, pf_b_idx;

    logic                             pf_issue;
    logic                             pf_last_land;
    logic                             fetch_ready;

    assign last_chunk   = (chunk_cnt == k_chunks_q - 1'b1);
    assign pf_issue     = pf_active && (pf_cnt < K);
    assign pf_last_land = pf_land_valid && (pf_land_cnt == K-1);
    // next chunk's staging is (or becomes this cycle) complete
    assign fetch_ready  = pf_done || pf_last_land;

    //----------------------------------------------------------------
    // Chunk handshake windows (see header for the timing contract)
    //----------------------------------------------------------------
    assign chunk_req = (state == S_IDLE && go)
                    || (state == S_FETCH && fetch_ready && !last_chunk);

    //----------------------------------------------------------------
    // Process 1 (clocked): state register, walk counters, command
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_INIT;
            beat_cnt   <= '0;
            settle_cnt <= '0;
            drain_cnt  <= '0;
            chunk_cnt  <= '0;
        end else begin
            state <= state_next;

            // counters: each counts only in its own state,
            // and resets on ENTRY to that state
            if (state_next == S_FEED)
                beat_cnt <= (state == S_FEED) ? beat_cnt + 1'b1 : '0;

            if (state_next == S_SETTLE)
                settle_cnt <= (state == S_SETTLE) ? settle_cnt + 1'b1 : '0;

            if (state_next == S_DRAIN)
                drain_cnt <= (state == S_DRAIN) ? drain_cnt + 1'b1 : '0;

            // chunk counter: 0 at walk start, +1 on each FEED->FETCH
            // loop-back (i.e. each completed non-final chunk)
            if (state == S_IDLE && go)
                chunk_cnt <= '0;
            else if (state == S_FEED && state_next == S_FETCH)
                chunk_cnt <= chunk_cnt + 1'b1;

            // walk-length capture: exactly once, on the accepted go
            if (state == S_IDLE && go)
                k_chunks_q <= k_chunks;

            // index capture: once per chunk, in the chunk_req window
            if (chunk_req) begin
                pf_a_idx <= a_tile_idx;
                pf_b_idx <= b_tile_idx;
            end
        end
    end

    //----------------------------------------------------------------
    // PREFETCH sequencer (control regs, async reset). Ordering
    // matters at the swap edge, where the completion of prefetch c
    // and the start of prefetch c+1 coincide: the chunk_req block
    // comes last so its writes win (start the new prefetch).
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pf_active     <= 1'b0;
            pf_done       <= 1'b0;
            pf_bank       <= 1'b0;
            pf_cnt        <= '0;
            pf_land_valid <= 1'b0;
            pf_land_cnt   <= '0;
            active_bank   <= 1'b0;
        end else begin
            // landing pipe: 1 cycle behind the issue side
            // (tile_buffer's registered-read latency)
            pf_land_valid <= pf_issue;
            pf_land_cnt   <= pf_cnt[$clog2(K)-1:0];

            // issue counter
            if (pf_issue)
                pf_cnt <= pf_cnt + 1'b1;

            // completion: the chunk's last word lands this cycle
            if (pf_last_land) begin
                pf_done   <= 1'b1;
                pf_active <= 1'b0;
            end

            // bank swap on EVERY entry to FEED: the just-completed
            // shadow bank becomes active, the drained-of-use active
            // bank becomes the new shadow
            if (state_next == S_FEED && state != S_FEED) begin
                active_bank <= ~active_bank;
                pf_done     <= 1'b0;
            end

            // new prefetch each chunk_req window (overrides the
            // completion/swap writes above at the shared edge)
            if (chunk_req) begin
                pf_active <= 1'b1;
                pf_done   <= 1'b0;
                pf_cnt    <= '0;
                // land in the bank that is the SHADOW once this edge
                // settles: at the accepted go nothing swaps, so it is
                // ~active_bank; at a swap edge the old active bank is
                // about to become the shadow
                if (state == S_IDLE) pf_bank <= ~active_bank;
                else                 pf_bank <= active_bank;
            end
        end
    end

    //----------------------------------------------------------------
    // Landing (datapath, no reset): unpack each arriving 32-bit word
    // into the shadow bank's 4 staging bytes for beat pf_land_cnt.
    // Element-wise only (Icarus). Part-selects are unsigned;
    // same-width assigns into the signed staging bytes, so no
    // extension is involved.
    //----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (pf_land_valid) begin
            for (int e = 0; e < N; e++)
                a_tile[pf_bank][e][pf_land_cnt] <= rd_data_a[e*DATA_WIDTH +: DATA_WIDTH];
            for (int e = 0; e < N; e++)
                b_tile[pf_bank][pf_land_cnt][e] <= rd_data_b[e*DATA_WIDTH +: DATA_WIDTH];
        end
    end

`ifndef SYNTHESIS
    // structural-hazard tripwire: a prefetch landing must never
    // target the bank FEED reads (same class as drain/feed overlap)
    always @(posedge clk)
        if (rst_n && pf_land_valid && (pf_bank == active_bank))
            $error("controller: PREFETCH landing into ACTIVE bank %0d", active_bank);
`endif

    //----------------------------------------------------------------
    // Process 2 (combinational): next-state logic
    //----------------------------------------------------------------
    always_comb begin
        state_next = state;                       // default: stay

        unique case (state)
            S_INIT:   state_next = S_IDLE;

            S_IDLE:   if (go) state_next = S_FETCH;

            // remainder wait: leave as soon as the prefetch for the
            // upcoming chunk is complete (cold walk start: all K+1
            // prefetch cycles are spent here; mid-walk: exactly the
            // 1 landing cycle FEED could not hide)
            S_FETCH:  if (fetch_ready)
                          state_next = S_FEED;

            // K-walk loop-back: more chunks -> back to FETCH for the
            // remainder. THE ACCUMULATORS INTENTIONALLY CARRY PARTIAL
            // SUMS ACROSS CHUNKS: this path must never pass through
            // any state or logic that asserts clear or drain (it
            // doesn't - clear is S_INIT-only, drain is S_DRAIN-only).
            // No SETTLE between chunks either: the 1-cycle gap in the
            // valid stream is handled by valid gating (tb_pe_os T2).
            // SETTLE runs once, after the FINAL chunk's FEED.
            // (A prefetch needs K+1 cycles vs K FEED cycles, so it is
            // never already complete here - the remainder is >= 1.)
            S_FEED:   if (beat_cnt == K-1) begin
                          if (last_chunk) state_next = S_SETTLE;
                          else            state_next = S_FETCH;
                      end

            S_SETTLE: if (settle_cnt == SETTLE_CYCLES-1)
                          state_next = S_DRAIN;

            S_DRAIN:  if (drain_cnt == N-1)
                          state_next = S_IDLE;

            default:  state_next = S_INIT;        // X/illegal recovery
        endcase
    end

    //----------------------------------------------------------------
    // Tile-buffer read ports: driven ONLY by the prefetch sequencer
    // (single read port per buffer, shared by construction - the old
    // S_FETCH issue side no longer exists)
    //----------------------------------------------------------------
    assign rd_en_a   = pf_issue;
    assign rd_en_b   = pf_issue;
    assign rd_addr_a = pf_a_idx * K + pf_cnt;
    assign rd_addr_b = pf_b_idx * K + pf_cnt;

    //----------------------------------------------------------------
    // Outputs (Moore: pure functions of state + counters)
    //----------------------------------------------------------------
    assign clear        = (state == S_INIT);
    assign busy         = (state != S_IDLE);
    assign drain        = (state == S_DRAIN);
    assign result_valid = (state == S_DRAIN);
    assign result_row   = $clog2(N)'(N-1) - drain_cnt;
    assign valid_feed   = (state == S_FEED);

    // perf flags: explicit Moore exports, never the raw enum
    assign in_fetch     = (state == S_FETCH);
    assign in_feed      = (state == S_FEED);
    assign in_settle    = (state == S_SETTLE);
    assign in_drain     = (state == S_DRAIN);
    assign in_idle      = (state == S_IDLE);
    assign chunk_done   = (state == S_FEED  && beat_cnt  == K-1);
    assign tile_done    = (state == S_DRAIN && drain_cnt == N-1);

    for (genvar i = 0; i < N; i++) begin : gen_feed
        assign a_feed[i] = (state == S_FEED) ? a_tile[active_bank][i][beat_cnt] : '0;
        assign b_feed[i] = (state == S_FEED) ? b_tile[active_bank][beat_cnt][i] : '0;
    end

endmodule
