`timescale 1ns / 1ps
//====================================================================
// tiling_engine: computes tile-multiple matmuls (M, K, N each a
// multiple of 4) by walking 4x4 tiles over matmul_unit v4.
//   For each output tile (I,J): one go with k_chunks=k_tiles, the
//   K-walk (G) serviced through chunk_req, PEs accumulate across
//   chunks, one drain, and the finished C-tile rows are captured
//   into the output buffer.
//
//   Ragged sizes (M/K/N not multiples of 4) are OUT OF SCOPE here:
//   zero-padding is the responsibility of whoever loads the buffers.
//
//   Tile layouts (indices into the respective buffers):
//     A tiles row-major : a_tile_idx = I*k_tiles + G
//     B tiles           : b_tile_idx = G*n_tiles + J
//     C tiles           : c_tile_slot = I*n_tiles + J
//   Output buffer word = one drain row (4 x 32-bit lanes, lane 0 =
//   column 0 in bits [31:0]); word address = c_tile_slot*4 +
//   result_row. Capacity: problems must fit the buffers, i.e.
//   m_tiles*k_tiles and k_tiles*n_tiles <= AB_DEPTH/K tiles, and
//   m_tiles*n_tiles <= OUT_DEPTH/N C-tile slots.
//
//   Loop order: I outer, J middle, G innermost. G MUST be innermost:
//   the K-walk's partial sums live in the PE accumulators, so a
//   whole walk has to finish (and drain) before moving to another
//   output tile.
//
//   Strictly sequential per output tile (issue go, service
//   chunk_req and result_valid, wait for !busy, advance). FUTURE
//   OPTIMIZATION: overlap the next tile's go with the previous
//   drain/collect - the handshakes already permit it.
//====================================================================
module tiling_engine #(
    parameter int N          = 4,
    parameter int K          = 4,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32,
    parameter int CHUNK_W    = 4,
    parameter int AB_DEPTH   = 64,   // a/b buffer beat-words (16 tiles each)
    parameter int OUT_DEPTH  = 64    // output buffer words (16 C-tiles at 4 words each)
)(
    input  logic clk,
    input  logic rst_n,

    // command
    input  logic                            start,
    input  logic [3:0]                      m_tiles,   // problem size in tiles, >= 1
    input  logic [3:0]                      n_tiles,
    input  logic [3:0]                      k_tiles,
    output logic                            done,      // 1-cycle pulse at completion
    output logic                            busy,

    // pass-through write ports for the A/B tile buffers (host/TB side)
    input  logic                            wr_en_a,
    input  logic [$clog2(AB_DEPTH)-1:0]     wr_addr_a,
    input  logic [31:0]                     wr_data_a,
    input  logic                            wr_en_b,
    input  logic [$clog2(AB_DEPTH)-1:0]     wr_addr_b,
    input  logic [31:0]                     wr_data_b,

    // output buffer read port (host/TB side; 1-cycle read latency)
    input  logic                            out_rd_en,
    input  logic [$clog2(OUT_DEPTH)-1:0]    out_rd_addr,
    output logic [N*ACC_WIDTH-1:0]          out_rd_data
);

    localparam int OUT_WORD_WIDTH = N * ACC_WIDTH;   // one drain row per word
    localparam int TW = $clog2(AB_DEPTH/K);

    //----------------------------------------------------------------
    // Engine FSM
    //----------------------------------------------------------------
    typedef enum logic [1:0] {
        T_IDLE,
        T_GO,       // one cycle: go high, chunk 0 (G=0) indices sampled
        T_BUSY,     // service chunk_req + drain capture, wait !mm_busy
        T_DONE      // one-cycle done pulse
    } t_state_t;

    t_state_t t_state, t_next;

    // latched problem size and tile walk counters
    logic [3:0]         m_q, n_q, k_q;
    logic [3:0]         tile_i, tile_j;
    logic [CHUNK_W-1:0] g_cnt;
    logic               last_tile;

    // matmul_unit interface nets
    logic                        mm_go, mm_busy, mm_chunk_req;
    logic [TW-1:0]               mm_a_idx, mm_b_idx;
    logic                        mm_result_valid;
    logic [$clog2(N)-1:0]        mm_result_row;
    logic signed [ACC_WIDTH-1:0] mm_drain [N];
    logic [CHUNK_W-1:0]          mm_k_chunks;

    // output-buffer write-side nets
    logic                            c_wr_en;
    logic [$clog2(OUT_DEPTH)-1:0]    c_wr_addr;
    logic [OUT_WORD_WIDTH-1:0]       c_wr_data;

    assign last_tile = (tile_i == m_q - 1'b1) && (tile_j == n_q - 1'b1);

    //----------------------------------------------------------------
    // Clocked: state, size latching, tile/chunk counters
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            t_state <= T_IDLE;
            m_q     <= '0;
            n_q     <= '0;
            k_q     <= '0;
            tile_i  <= '0;
            tile_j  <= '0;
            g_cnt   <= '0;
        end else begin
            t_state <= t_next;

            // size capture + walk restart: on the accepted start
            if (t_state == T_IDLE && start) begin
                m_q    <= m_tiles;
                n_q    <= n_tiles;
                k_q    <= k_tiles;
                tile_i <= '0;
                tile_j <= '0;
            end

            // advance (I,J) when the current output tile finishes
            // (J middle, I outer; G is handled below, innermost)
            if (t_state == T_BUSY && !mm_busy) begin
                if (tile_j == n_q - 1'b1) begin
                    tile_j <= '0;
                    tile_i <= tile_i + 1'b1;
                end else begin
                    tile_j <= tile_j + 1'b1;
                end
            end

            // G counter: 0 on entry to T_GO; +1 as each chunk_req
            // window closes (the controller registered G's indices
            // at that same edge, so g_cnt always points at the NEXT
            // chunk to present)
            if (t_next == T_GO && t_state != T_GO)
                g_cnt <= '0;
            else if (mm_chunk_req)
                g_cnt <= g_cnt + 1'b1;
        end
    end

    //----------------------------------------------------------------
    // Next-state
    //----------------------------------------------------------------
    always_comb begin
        t_next = t_state;

        unique case (t_state)
            T_IDLE: if (start) t_next = T_GO;

            T_GO:   t_next = T_BUSY;

            // mm_busy falls the cycle AFTER the last drain row was
            // written, so all N rows of this C-tile are in the
            // output buffer by the time we advance
            T_BUSY: if (!mm_busy) begin
                        if (last_tile) t_next = T_DONE;
                        else           t_next = T_GO;
                    end

            T_DONE: t_next = T_IDLE;

            default: t_next = T_IDLE;
        endcase
    end

    //----------------------------------------------------------------
    // matmul_unit command side (Moore). Indices are presented
    // combinationally from the walk counters while chunk_req is
    // high; the controller registers them at the posedge ending
    // that cycle. Small multiplies synthesize to LUTs fine.
    //----------------------------------------------------------------
    assign mm_go       = (t_state == T_GO);
    assign mm_k_chunks = k_q[CHUNK_W-1:0];
    assign mm_a_idx    = tile_i * k_q + g_cnt;
    assign mm_b_idx    = g_cnt * n_q + tile_j;

    assign busy = (t_state != T_IDLE);
    assign done = (t_state == T_DONE);

    //----------------------------------------------------------------
    // Drain capture: each result_valid cycle carries one finished
    // C-tile row on the 4 drain lanes; pack lane 0 into bits [31:0]
    // and write one 128-bit word per cycle (no serialization)
    //----------------------------------------------------------------
    assign c_wr_en   = mm_result_valid;
    assign c_wr_addr = (tile_i * n_q + tile_j) * N + mm_result_row;
    assign c_wr_data = {mm_drain[3], mm_drain[2], mm_drain[1], mm_drain[0]};

    //----------------------------------------------------------------
    // Instances
    //----------------------------------------------------------------
    matmul_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(AB_DEPTH), .CHUNK_W(CHUNK_W)
    ) u_mm (
        .clk          (clk),
        .rst_n        (rst_n),
        .go           (mm_go),
        .k_chunks     (mm_k_chunks),
        .chunk_req    (mm_chunk_req),
        .a_tile_idx   (mm_a_idx),
        .b_tile_idx   (mm_b_idx),
        .wr_en_a      (wr_en_a),
        .wr_addr_a    (wr_addr_a),
        .wr_data_a    (wr_data_a),
        .wr_en_b      (wr_en_b),
        .wr_addr_b    (wr_addr_b),
        .wr_data_b    (wr_data_b),
        .busy         (mm_busy),
        .result_valid (mm_result_valid),
        .result_row   (mm_result_row),
        .drain_out    (mm_drain)
    );

    tile_buffer #(
        .DEPTH(OUT_DEPTH), .WORD_WIDTH(OUT_WORD_WIDTH)
    ) u_buf_c (
        .clk     (clk),
        .wr_en   (c_wr_en),
        .wr_addr (c_wr_addr),
        .wr_data (c_wr_data),
        .rd_en   (out_rd_en),
        .rd_addr (out_rd_addr),
        .rd_data (out_rd_data)
    );

endmodule
