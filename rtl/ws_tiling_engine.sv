`timescale 1ns / 1ps
//====================================================================
// ws_tiling_engine v1 (tiling spec step 1): sweeps weight tiles
// N-outer / K-inner over ws_quant_unit, accumulating raw INT32
// partials in per-column banks. Loop order rationale (spec section
// 1): WS has no persistent accumulator across weight-tile swaps, so
// K-sweep partials live HERE; N-outer means only one N-tile's worth
// of accumulator state ever exists.
//
//   Sequencing: one inner go / wait-done per (n_tile, k_tile) pair.
//   Weight tile (n,k) is read from buffer tile index n*k_tiles + k
//   (t_idx just increments across the sweep); activation chunk k is
//   read from a_base = k*n_vecs. The engine does NOT source tile
//   data - w_tile_idx/a_base tell the loader (TB now, DMA later)
//   which tile is in play, mirroring OS's chunk_req pattern.
//
//   Accumulator: N banks x MAX_N_VECS x ACC_WIDTH, accumulate-on-
//   write (k_tile 0 overwrites, later k-tiles add). Banks and their
//   adders are fabric (0 DSPs expected on sequencing/accumulate).
//   Overflow argument: the capacity guard bounds total K at DEPTH
//   rows, so |acc| <= DEPTH * 127 * 127 < 2^21 - INT32 banks cannot
//   overflow at any accepted configuration. Revisit if DEPTH grows
//   past ~2^16 or ACC_WIDTH shrinks.
//
//   cfg_err (sticky, cleared by the next ACCEPTED go): a go with
//   n_vecs == 0, n_vecs > MAX_N_VECS, k_tiles == 0, n_tiles == 0,
//   k_tiles*n_tiles > DEPTH/K, or k_tiles*n_vecs > DEPTH is REJECTED
//   - busy never rises. Surfaced as STATUS[2] by axi_lite_ctrl.
//   Note: at the default MAX_N_VECS=16 with VEC_W=4 the n_vecs >
//   MAX_N_VECS leg is unreachable (register max is 15); it arms only
//   when an integration sets MAX_N_VECS below 2^VEC_W - 1. n_vecs=0
//   is rejected and does NOT mean 2^VEC_W.
//
//   HOST CONTRACT (tiling spec section 3): during tiled operation,
//   ws_quant_unit's own INT8 stream (q8_out/q8_valid) keeps firing
//   on every inner tile and requantizes PER-TILE PARTIAL sums - it
//   is NOT authoritative when K_TILES > 1 and must be ignored by
//   anything downstream. Only this engine's res_* stream carries
//   final results (per N-tile: v outer 0..n_vecs-1, j inner
//   0..N-1, one value per cycle, 2-cycle pipe latency; global
//   column = res_ntile*N + res_col). For K_TILES == 1 both streams
//   are numerically correct; res_* remains the single contract.
//   The final tap is the THIRD instance of the shared requant core
//   (no fork), fired once per N-tile after its K-sweep completes.
//====================================================================
module ws_tiling_engine #(
    parameter int N           = 4,
    parameter int K           = 4,
    parameter int ACC_WIDTH   = 32,
    parameter int OUT_WIDTH   = 8,
    parameter int DEPTH       = 64,
    parameter int VEC_W       = 4,
    parameter int TILE_W      = 4,
    parameter int MAX_N_VECS  = 16,
    parameter int SCALE_WIDTH = 16,
    parameter int SHIFT_WIDTH = 5
)(
    input  logic clk,
    input  logic rst_n,

    // outer command (axi_lite_ctrl's core side drives this)
    input  logic              go,
    input  logic [VEC_W-1:0]  n_vecs,
    input  logic [TILE_W-1:0] k_tiles,
    input  logic [TILE_W-1:0] n_tiles,
    output logic              busy,
    output logic              cfg_err,

    // per-tensor requant config (feeds the step-2 final tap)
    input  logic                          relu_en,
    input  logic signed [SCALE_WIDTH-1:0] scale_mult,
    input  logic [SHIFT_WIDTH-1:0]        shift_amt,

    // inner command to ws_quant_unit
    output logic                          core_go,
    input  logic                          core_busy,
    output logic [VEC_W-1:0]              core_n_vecs,
    output logic [$clog2(DEPTH/K)-1:0]    w_tile_idx,
    output logic [$clog2(DEPTH)-1:0]      a_base,

    // raw bottom-edge tap (column-skewed, gated by sum_valid_in)
    input  logic signed [ACC_WIDTH-1:0]   sum_in       [N],
    input  logic                          sum_valid_in [N],

    // final results (one per (vec, col) per N-tile, after its K-sweep)
    output logic                          res_valid,
    output logic signed [OUT_WIDTH-1:0]   res_q8,
    output logic [VEC_W-1:0]              res_vec,
    output logic [$clog2(N)-1:0]          res_col,
    output logic [TILE_W-1:0]             res_ntile
);

    localparam int TW = $clog2(DEPTH/K);
    localparam int AW = $clog2(DEPTH);

    //----------------------------------------------------------------
    // Engine FSM
    //----------------------------------------------------------------
    typedef enum logic [2:0] {
        E_IDLE,
        E_GO,      // one cycle: inner go pulse
        E_WAITB,   // wait for core busy to rise (1 cycle after go)
        E_RUN,     // collect valids, wait for core busy to fall
        E_ADV,     // advance k, or enter the drain
        E_DRAIN,   // walk banks: v outer, j inner, one value/cycle
        E_FLUSH    // 2 cycles: let the read->requant pipe emit
    } e_state_t;

    e_state_t state;

    // latched command + walk counters (control: async reset)
    logic [VEC_W-1:0]  nv_q;
    logic [TILE_W-1:0] kt_q, nt_q;
    logic [TILE_W-1:0] k_cnt, n_cnt;
    logic [TW-1:0]     t_idx;
    logic [AW-1:0]     a_base_r;

    // drain walk (control: async reset)
    logic [VEC_W-1:0]        dr_v;
    logic [$clog2(N)-1:0]    dr_j;
    logic                    flush_cnt;

    // configuration validation (combinational, evaluated at go)
    logic cfg_bad;
    logic [2*TILE_W-1:0]      kt_x_nt;
    logic [TILE_W+VEC_W-1:0]  kt_x_nv;

    assign kt_x_nt = k_tiles * n_tiles;
    assign kt_x_nv = k_tiles * n_vecs;
    assign cfg_bad = (n_vecs == '0) || (n_vecs > MAX_N_VECS)
                  || (k_tiles == '0) || (n_tiles == '0)
                  || (kt_x_nt > DEPTH/K) || (kt_x_nv > DEPTH);

    logic k_first;
    assign k_first = (k_cnt == '0);

    logic collecting;
    assign collecting = (state == E_WAITB) || (state == E_RUN);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= E_IDLE;
            nv_q     <= '0;
            kt_q     <= '0;
            nt_q     <= '0;
            k_cnt    <= '0;
            n_cnt    <= '0;
            t_idx    <= '0;
            a_base_r <= '0;
            cfg_err  <= 1'b0;
            dr_v     <= '0;
            dr_j     <= '0;
            flush_cnt <= 1'b0;
        end else begin
            unique case (state)
                E_IDLE: begin
                    if (go) begin
                        if (cfg_bad) begin
                            cfg_err <= 1'b1;
                        end else begin
                            cfg_err  <= 1'b0;
                            nv_q     <= n_vecs;
                            kt_q     <= k_tiles;
                            nt_q     <= n_tiles;
                            k_cnt    <= '0;
                            n_cnt    <= '0;
                            t_idx    <= '0;
                            a_base_r <= '0;
                            state    <= E_GO;
                        end
                    end
                end

                E_GO:    state <= E_WAITB;

                E_WAITB: if (core_busy) state <= E_RUN;

                E_RUN:   if (!core_busy) state <= E_ADV;

                E_ADV: begin
                    if (k_cnt == kt_q - 1'b1) begin
                        dr_v  <= '0;                     // K-sweep done:
                        dr_j  <= '0;                     // drain this N-tile
                        state <= E_DRAIN;
                    end else begin
                        k_cnt    <= k_cnt + 1'b1;        // next K-tile
                        t_idx    <= t_idx + 1'b1;
                        a_base_r <= a_base_r + nv_q;
                        state    <= E_GO;
                    end
                end

                E_DRAIN: begin
                    if (dr_j == N - 1) begin
                        dr_j <= '0;
                        if (dr_v == nv_q - 1'b1) begin
                            flush_cnt <= 1'b0;
                            state     <= E_FLUSH;
                        end else begin
                            dr_v <= dr_v + 1'b1;
                        end
                    end else begin
                        dr_j <= dr_j + 1'b1;
                    end
                end

                E_FLUSH: begin
                    flush_cnt <= 1'b1;
                    if (flush_cnt) begin
                        if (n_cnt == nt_q - 1'b1) begin
                            state <= E_IDLE;             // sweep done
                        end else begin
                            n_cnt    <= n_cnt + 1'b1;    // next N-tile
                            k_cnt    <= '0;
                            t_idx    <= t_idx + 1'b1;
                            a_base_r <= '0;
                            state    <= E_GO;
                        end
                    end
                end

                default: state <= E_IDLE;                // X recovery
            endcase
        end
    end

    //----------------------------------------------------------------
    // Outputs (Moore)
    //----------------------------------------------------------------
    assign busy        = (state != E_IDLE);
    assign core_go     = (state == E_GO);
    assign core_n_vecs = nv_q;
    assign w_tile_idx  = t_idx;
    assign a_base      = a_base_r;

    //----------------------------------------------------------------
    // Accumulator banks: N columns x MAX_N_VECS vectors, accumulate-
    // on-write. acc/adders are datapath: NO reset (k_first overwrite
    // covers post-reset X). Each generate column writes only its own
    // acc[j] row and v_cnt[j] (distinct longest static prefixes).
    //----------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] acc [N][MAX_N_VECS];
    logic [VEC_W-1:0]            v_cnt [N];

    for (genvar j = 0; j < N; j++) begin : gen_acc
        // per-vector write index (control: async reset)
        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n)
                v_cnt[j] <= '0;
            else if (core_go)
                v_cnt[j] <= '0;
            else if (collecting && sum_valid_in[j])
                v_cnt[j] <= v_cnt[j] + 1'b1;
        end

        // bank write (datapath: no reset)
        always_ff @(posedge clk) begin
            if (collecting && sum_valid_in[j]) begin
                if (k_first)
                    acc[j][v_cnt[j]] <= sum_in[j];
                else
                    acc[j][v_cnt[j]] <= acc[j][v_cnt[j]] + sum_in[j];
            end
        end
    end

    //----------------------------------------------------------------
    // Drain pipe: bank read (stage 1) -> shared requant core ->
    // registered result stream (stage 2). One value per E_DRAIN
    // cycle; res_valid trails the issue by 2 cycles (E_FLUSH covers
    // the tail, so busy stays high until the last result is out).
    // Valids are control (async reset); data regs are datapath (none).
    //----------------------------------------------------------------
    logic                        dr_issue;
    assign dr_issue = (state == E_DRAIN);

    logic                        p1_valid;
    logic signed [ACC_WIDTH-1:0] p1_acc;
    logic [VEC_W-1:0]            p1_v;
    logic [$clog2(N)-1:0]        p1_j;
    logic [TILE_W-1:0]           p1_nt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) p1_valid <= 1'b0;
        else        p1_valid <= dr_issue;
    end

    always_ff @(posedge clk) begin
        p1_acc <= acc[dr_j][dr_v];
        p1_v   <= dr_v;
        p1_j   <= dr_j;
        p1_nt  <= n_cnt;
    end

    logic signed [OUT_WIDTH-1:0] rq_final;

    requant #(
        .ACC_WIDTH(ACC_WIDTH), .OUT_WIDTH(OUT_WIDTH),
        .SCALE_WIDTH(SCALE_WIDTH), .SHIFT_WIDTH(SHIFT_WIDTH)
    ) u_rq_final (
        .acc_in     (p1_acc),
        .relu_en    (relu_en),
        .scale_mult (scale_mult),
        .shift_amt  (shift_amt),
        .out        (rq_final)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) res_valid <= 1'b0;
        else        res_valid <= p1_valid;
    end

    always_ff @(posedge clk) begin
        res_q8    <= rq_final;
        res_vec   <= p1_v;
        res_col   <= p1_j;
        res_ntile <= p1_nt;
    end

endmodule
