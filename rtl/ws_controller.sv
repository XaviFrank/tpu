`timescale 1ns / 1ps
//====================================================================
// ws_controller v1 (spec step 3): tile-lifecycle FSM for the
// weight-stationary array. LOAD_WEIGHTS/STREAM skeleton, single
// buffered: BRAM reads feed the array's top edge directly (the
// double-buffered weight prefetch was dropped as obsolete-by-
// construction - see docs/ws_design_notes.md).
//
//   go (w_tile_idx, a_base, n_vecs) ->
//     W_LOAD  : N weight-buffer reads, tile rows fed BOTTOM-FIRST
//               straight onto the top edge. The registered-read
//               latency folds into the pipeline: read row K-1-t at
//               cycle t, it lands and is fed at t+1. One broadcast
//               w_capture pulse on the cycle every row's weight sits
//               aligned in its column (bottom-first feeding aligns
//               all rows simultaneously - see ws_pe). Duration
//               N+2 cycles: 1 landing lead-in + N shifts + 1 capture.
//               (Certified at N=4 by tb_ws_controller; N=16 by
//               tb_ws_n16.)
//     W_STREAM: n_vecs activation words read back-to-back from the
//               activation buffer (addr a_base+v) and fed through
//               input_skew, one vector per cycle. Duration n_vecs+1
//               (landing lead-in again).
//     W_FLUSH : 2N+1 cycles; the last vector's column-(N-1) result
//               appears on the last flush cycle, so every result is
//               on the wires before busy falls. There is no SETTLE
//               and no DRAIN: results exit continuously at the
//               bottom edge, gated by the array's own sum_valid.
//   -> W_IDLE. (The spec sketch's INIT(clear) state was dropped: WS
//   has no persistent accumulator to clear, and valid gating covers
//   post-reset X in the sum path - flagged deviation.)
//
//   Word width: one buffer word = one tile row = one activation
//   vector = N packed int8s (N*DATA_WIDTH bits; 32 at N=4, 128 at
//   N=16). K = words per weight tile = N rows; keep K a power of
//   two (load_row's all-ones subtraction relies on it).
//
//   Result convention: bottom of column j = sum_i w[i][j]*a[i],
//   i.e. y = W_loaded^T * a; hosts store weight tiles transposed.
//   b_top's upper bits are driven 0 during LOAD (capture uses the
//   low byte only; the junk that shifts on afterward never has a
//   valid attached).
//====================================================================
module ws_controller #(
    parameter int N          = 4,
    parameter int K          = 4,   // words per weight tile (= N rows)
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32,
    parameter int DEPTH      = 64,  // buffer beat-words; DEPTH/K tiles
    parameter int VEC_W      = 4    // width of n_vecs (n_vecs >= 1)
)(
    input  logic clk,
    input  logic rst_n,

    // command interface
    input  logic                          go,
    input  logic [$clog2(DEPTH/K)-1:0]    w_tile_idx,
    input  logic [$clog2(DEPTH)-1:0]      a_base,    // first activation word
    input  logic [VEC_W-1:0]              n_vecs,
    output logic                          busy,

    // weight-buffer read port (rd_data valid 1 cycle after rd_en)
    output logic                          rd_en_w,
    output logic [$clog2(DEPTH)-1:0]      rd_addr_w,
    input  logic [N*DATA_WIDTH-1:0]       rd_data_w,

    // activation-buffer read port
    output logic                          rd_en_a,
    output logic [$clog2(DEPTH)-1:0]      rd_addr_a,
    input  logic [N*DATA_WIDTH-1:0]       rd_data_a,

    // datapath control (drives ws_array / input_skew)
    output logic                          load_mode,
    output logic                          w_capture,
    output logic                          valid_feed,
    output logic signed [DATA_WIDTH-1:0]  a_feed [N],
    output logic signed [ACC_WIDTH-1:0]   b_top  [N]
);

    //----------------------------------------------------------------
    // State encoding
    //----------------------------------------------------------------
    typedef enum logic [1:0] {
        W_IDLE,
        W_LOAD,     // weight shift-in + capture
        W_STREAM,   // activation feeding
        W_FLUSH     // in-flight psums exit the array
    } state_t;

    state_t state, state_next;

    //----------------------------------------------------------------
    // Cycle budgets (certified by TB at N=4 and N=16 - see header)
    //----------------------------------------------------------------
    localparam int LOAD_CYCLES  = N + 2;
    localparam int FLUSH_CYCLES = 2*N + 1;
    localparam int KW           = $clog2(K);

    logic [$clog2(LOAD_CYCLES)-1:0]  load_cnt;
    logic [VEC_W-1:0]                stream_cnt;
    logic [$clog2(FLUSH_CYCLES)-1:0] flush_cnt;

    // latched command
    logic [$clog2(DEPTH/K)-1:0] w_idx_q;
    logic [$clog2(DEPTH)-1:0]   a_base_q;
    logic [VEC_W-1:0]           n_vecs_q;

    // landing flags: read data is on rd_data_* one cycle after rd_en
    logic land_w, land_a;

    // bottom-first row index for the weight fetch (K is a power of
    // two, so K-1 is all-ones and the subtraction never underflows
    // while rd_en_w is high)
    logic [KW-1:0] load_row;
    assign load_row = {KW{1'b1}} - load_cnt[KW-1:0];

    //----------------------------------------------------------------
    // Process 1 (clocked): state, counters, command latch
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= W_IDLE;
            load_cnt   <= '0;
            stream_cnt <= '0;
            flush_cnt  <= '0;
            land_w     <= 1'b0;
            land_a     <= 1'b0;
        end else begin
            state <= state_next;

            // counters: each counts only in its own state,
            // and resets on ENTRY to that state
            if (state_next == W_LOAD)
                load_cnt <= (state == W_LOAD) ? load_cnt + 1'b1 : '0;

            if (state_next == W_STREAM)
                stream_cnt <= (state == W_STREAM) ? stream_cnt + 1'b1 : '0;

            if (state_next == W_FLUSH)
                flush_cnt <= (state == W_FLUSH) ? flush_cnt + 1'b1 : '0;

            // landing pipes (registered-read latency)
            land_w <= rd_en_w;
            land_a <= rd_en_a;

            // command capture: exactly once, on the accepted go
            if (state == W_IDLE && go) begin
                w_idx_q  <= w_tile_idx;
                a_base_q <= a_base;
                n_vecs_q <= n_vecs;
            end
        end
    end

    //----------------------------------------------------------------
    // Process 2 (combinational): next-state logic
    //----------------------------------------------------------------
    always_comb begin
        state_next = state;                       // default: stay

        unique case (state)
            W_IDLE:   if (go) state_next = W_LOAD;

            W_LOAD:   if (load_cnt == LOAD_CYCLES-1)
                          state_next = W_STREAM;

            // exit on the last landing cycle (feed n_vecs lands on
            // stream_cnt == n_vecs, one behind the last issue)
            W_STREAM: if (stream_cnt == n_vecs_q)
                          state_next = W_FLUSH;

            W_FLUSH:  if (flush_cnt == FLUSH_CYCLES-1)
                          state_next = W_IDLE;

            default:  state_next = W_IDLE;        // X/illegal recovery
        endcase
    end

    //----------------------------------------------------------------
    // Buffer read ports
    //----------------------------------------------------------------
    assign rd_en_w   = (state == W_LOAD) && (load_cnt < N);
    assign rd_addr_w = w_idx_q * K + load_row;

    assign rd_en_a   = (state == W_STREAM) && (stream_cnt < n_vecs_q);
    assign rd_addr_a = a_base_q + stream_cnt;

    //----------------------------------------------------------------
    // Outputs (Moore: pure functions of state + counters)
    //----------------------------------------------------------------
    assign busy       = (state != W_IDLE);
    assign load_mode  = (state == W_LOAD);
    assign w_capture  = (state == W_LOAD) && (load_cnt == LOAD_CYCLES-1);
    assign valid_feed = (state == W_STREAM) && land_a;

    for (genvar e = 0; e < N; e++) begin : gen_feed
        // weight word lane e -> column e's top edge, low byte only
        assign b_top[e] = ((state == W_LOAD) && land_w)
            ? {{(ACC_WIDTH-DATA_WIDTH){1'b0}}, rd_data_w[e*DATA_WIDTH +: DATA_WIDTH]}
            : '0;
        // activation word lane e -> row e's feed
        assign a_feed[e] = ((state == W_STREAM) && land_a)
            ? rd_data_a[e*DATA_WIDTH +: DATA_WIDTH]
            : '0;
    end

endmodule
