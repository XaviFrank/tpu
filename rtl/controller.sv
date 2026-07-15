`timescale 1ns / 1ps
//====================================================================
// controller: tile-lifecycle FSM for matmul_unit
//   go -> latch tile -> FEED (K beats) -> SETTLE (2N+1) -> DRAIN (N)
//   Strictly sequential v1: no feed/drain overlap (costs 1 cycle/tile
//   at N=4; overlap engine is a ROADMAP item, break-even ~N>=8).
//====================================================================
module controller #(
    parameter int N          = 4,
    parameter int K          = 4,
    parameter int DATA_WIDTH = 8
)(
    input  logic clk,
    input  logic rst_n,

    // command interface
    input  logic                          go,
    input  logic signed [DATA_WIDTH-1:0]  a_tile_in [N][K],
    input  logic signed [DATA_WIDTH-1:0]  b_tile_in [K][N],
    output logic                          busy,

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
        S_FEED,
        S_SETTLE,
        S_DRAIN
    } state_t;

    state_t state, state_next;

    //----------------------------------------------------------------
    // Counters and tile buffers
    //----------------------------------------------------------------
    localparam int SETTLE_CYCLES = 2*N; // boundary certified, 2N-1 clips the boundary

    logic [$clog2(K)-1:0]             beat_cnt;
    logic [$clog2(SETTLE_CYCLES)-1:0] settle_cnt;
    logic [$clog2(N)-1:0]             drain_cnt;

    logic signed [DATA_WIDTH-1:0] a_tile [N][K];
    logic signed [DATA_WIDTH-1:0] b_tile [K][N];

    //----------------------------------------------------------------
    // Process 1 (clocked): state register, counters, tile latching
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_INIT;
            beat_cnt   <= '0;
            settle_cnt <= '0;
            drain_cnt  <= '0;
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

            // tile capture: exactly once, on the accepted go
            if (state == S_IDLE && go) begin
                for (int i = 0; i < N; i++)
                    for (int k = 0; k < K; k++) begin
                        a_tile[i][k] <= a_tile_in[i][k];
                        b_tile[k][i] <= b_tile_in[k][i];
                    end
            end
        end
    end

    //----------------------------------------------------------------
    // Process 2 (combinational): next-state logic
    //----------------------------------------------------------------
    always_comb begin
        state_next = state;                       // default: stay

        unique case (state)
            S_INIT:   state_next = S_IDLE;

            S_IDLE:   if (go) state_next = S_FEED;

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