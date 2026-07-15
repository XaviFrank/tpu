`timescale 1ns / 1ps
//====================================================================
// controller v4: tile-lifecycle FSM for matmul_unit, with K-walk
//   go (with k_chunks + first tile indices) ->
//     repeat k_chunks times: FETCH tiles from BRAM into staging,
//     FEED them (accumulating in the PEs) ->
//   one SETTLE -> one DRAIN -> IDLE
//
//   Chunk handshake: chunk_req is high for exactly ONE cycle per
//   chunk; a_tile_idx/b_tile_idx are registered at the posedge that
//   ENDS the chunk_req cycle, and that chunk's FETCH begins on the
//   following cycle. The windows are:
//     chunk 0   : the accepted-go cycle (state==S_IDLE && go)
//     chunk c>0 : the LAST FEED cycle of chunk c-1
//   The issuer presents indices combinationally while chunk_req is
//   high. k_chunks (>= 1; 0 is illegal) is latched once, at the
//   accepted go.
//
//   Perf flags (in_*, chunk_done, tile_done) are explicit Moore
//   outputs for perf_counters - pure observers, they feed no control
//   logic. The raw state enum is never exported. S_INIT (one cycle
//   after reset) is covered by no flag.
//
//   Strictly sequential: no fetch/feed/drain overlap (overlap engine
//   is a ROADMAP item). k_chunks=1 reproduces v3 exactly.
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
        S_FETCH,
        S_FEED,
        S_SETTLE,
        S_DRAIN
    } state_t;

    state_t state, state_next;

    //----------------------------------------------------------------
    // Counters, latched command, staging registers
    //----------------------------------------------------------------
    localparam int SETTLE_CYCLES = 2*N; // boundary certified, 2N-1 clips the boundary

    logic [$clog2(K)-1:0]             beat_cnt;
    logic [$clog2(SETTLE_CYCLES)-1:0] settle_cnt;
    logic [$clog2(N)-1:0]             drain_cnt;

    // fetch issue counter: 0..K-1 issue reads, K = landing-only cycle,
    // so it must count to K (one bit wider than beat_cnt)
    logic [$clog2(K+1)-1:0]           fetch_cnt;

    // landing side runs exactly 1 cycle behind the issue side
    // (tile_buffer's registered-read latency)
    logic                             land_valid;
    logic [$clog2(K)-1:0]             land_cnt;

    // K-walk bookkeeping: which chunk is in flight, out of how many
    logic [CHUNK_W-1:0]               chunk_cnt;
    logic [CHUNK_W-1:0]               k_chunks_q;
    logic                             last_chunk;

    logic [$clog2(DEPTH/K)-1:0]       a_idx_q, b_idx_q;

    logic signed [DATA_WIDTH-1:0] a_tile [N][K];
    logic signed [DATA_WIDTH-1:0] b_tile [K][N];

    assign last_chunk = (chunk_cnt == k_chunks_q - 1'b1);

    //----------------------------------------------------------------
    // Chunk handshake: one-cycle index-sampling window per chunk.
    // Window for chunk 0 is the accepted-go cycle; window for chunk
    // c>0 is the last FEED cycle of chunk c-1 (the same cycle the
    // FEED->FETCH loop-back is decided).
    //----------------------------------------------------------------
    assign chunk_req = (state == S_IDLE && go)
                    || (state == S_FEED && beat_cnt == K-1 && !last_chunk);

    //----------------------------------------------------------------
    // Process 1 (clocked): state register, counters, command latching
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_INIT;
            beat_cnt   <= '0;
            settle_cnt <= '0;
            drain_cnt  <= '0;
            fetch_cnt  <= '0;
            land_valid <= 1'b0;
            land_cnt   <= '0;
            chunk_cnt  <= '0;
        end else begin
            state <= state_next;

            // counters: each counts only in its own state,
            // and resets on ENTRY to that state
            if (state_next == S_FETCH)
                fetch_cnt <= (state == S_FETCH) ? fetch_cnt + 1'b1 : '0;

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

            // landing pipe: issue side asserts rd_en with fetch_cnt=c;
            // that word is on rd_data one cycle later, so delay the
            // count/valid by one cycle to know where it lands
            land_valid <= rd_en_a;
            land_cnt   <= fetch_cnt[$clog2(K)-1:0];

            // walk-length capture: exactly once, on the accepted go
            if (state == S_IDLE && go)
                k_chunks_q <= k_chunks;

            // index capture: once per chunk, in the chunk_req window
            // (covers the accepted go, since chunk_req is high there)
            if (chunk_req) begin
                a_idx_q <= a_tile_idx;
                b_idx_q <= b_tile_idx;
            end
        end
    end

    //----------------------------------------------------------------
    // Landing (datapath, no reset): unpack each arriving 32-bit word
    // into the 4 staging bytes for beat land_cnt. Element-wise only
    // (Icarus). Part-selects are unsigned; same-width assigns into
    // the signed staging bytes, so no extension is involved.
    //----------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (land_valid) begin
            for (int e = 0; e < N; e++)
                a_tile[e][land_cnt] <= rd_data_a[e*DATA_WIDTH +: DATA_WIDTH];
            for (int e = 0; e < N; e++)
                b_tile[land_cnt][e] <= rd_data_b[e*DATA_WIDTH +: DATA_WIDTH];
        end
    end

    //----------------------------------------------------------------
    // Process 2 (combinational): next-state logic
    //----------------------------------------------------------------
    always_comb begin
        state_next = state;                       // default: stay

        unique case (state)
            S_INIT:   state_next = S_IDLE;

            S_IDLE:   if (go) state_next = S_FETCH;

            // K+1 cycles total: fetch_cnt 0..K-1 issue reads, cycle K
            // is issue-idle and lets the last word land in staging
            // (the landing write and the FEED entry share an edge:
            // staging is stable before the first FEED beat uses it)
            S_FETCH:  if (fetch_cnt == K)
                          state_next = S_FEED;

            // K-walk loop-back: more chunks -> straight back to FETCH.
            // THE ACCUMULATORS INTENTIONALLY CARRY PARTIAL SUMS ACROSS
            // CHUNKS: this path must never pass through any state or
            // logic that asserts clear or drain (it doesn't - clear is
            // S_INIT-only, drain is S_DRAIN-only). No SETTLE between
            // chunks either: the FETCH gap in the valid stream is
            // handled by valid gating (verified in tb_pe_os T2).
            // SETTLE runs once, after the FINAL chunk's FEED.
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
    // Fetch issue side (Moore): read both buffers IN PARALLEL,
    // one beat per cycle, tile base + beat offset
    //----------------------------------------------------------------
    assign rd_en_a   = (state == S_FETCH) && (fetch_cnt < K);
    assign rd_en_b   = (state == S_FETCH) && (fetch_cnt < K);
    assign rd_addr_a = a_idx_q * K + fetch_cnt;
    assign rd_addr_b = b_idx_q * K + fetch_cnt;

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
        assign a_feed[i] = (state == S_FEED) ? a_tile[i][beat_cnt] : '0;
        assign b_feed[i] = (state == S_FEED) ? b_tile[beat_cnt][i] : '0;
    end

endmodule
