`timescale 1ns / 1ps
//====================================================================
// controller v3: tile-lifecycle FSM for matmul_unit
//   go (with tile indices) -> FETCH tiles from BRAM buffers into
//   staging regs -> FEED (K beats) -> SETTLE (2N) -> DRAIN (N)
//   Option A: fetch-then-feed; everything from FEED onward is
//   identical to v2. Strictly sequential: no fetch/feed/drain
//   overlap (overlap engine is a ROADMAP item).
//====================================================================
module controller #(
    parameter int N          = 4,
    parameter int K          = 4,
    parameter int DATA_WIDTH = 8,
    parameter int DEPTH      = 64   // tile-buffer beat-words; DEPTH/K tiles
)(
    input  logic clk,
    input  logic rst_n,

    // command interface (indices latched on the accepted go)
    input  logic                          go,
    input  logic [$clog2(DEPTH/K)-1:0]    a_tile_idx,
    input  logic [$clog2(DEPTH/K)-1:0]    b_tile_idx,
    output logic                          busy,

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
    output logic [$clog2(N)-1:0]          result_row
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
    // Counters, latched indices, staging registers
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

    logic [$clog2(DEPTH/K)-1:0]       a_idx_q, b_idx_q;

    logic signed [DATA_WIDTH-1:0] a_tile [N][K];
    logic signed [DATA_WIDTH-1:0] b_tile [K][N];

    //----------------------------------------------------------------
    // Process 1 (clocked): state register, counters, index latching
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

            // landing pipe: issue side asserts rd_en with fetch_cnt=c;
            // that word is on rd_data one cycle later, so delay the
            // count/valid by one cycle to know where it lands
            land_valid <= rd_en_a;
            land_cnt   <= fetch_cnt[$clog2(K)-1:0];

            // index capture: exactly once, on the accepted go
            if (state == S_IDLE && go) begin
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

            S_FEED:   if (beat_cnt == K-1)
                          state_next = S_SETTLE;

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

    for (genvar i = 0; i < N; i++) begin : gen_feed
        assign a_feed[i] = (state == S_FEED) ? a_tile[i][beat_cnt] : '0;
        assign b_feed[i] = (state == S_FEED) ? b_tile[beat_cnt][i] : '0;
    end

endmodule
