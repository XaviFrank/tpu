`timescale 1ns / 1ps
//====================================================================
// axi_lite_ctrl v3 (AXI spec step 4): full single-tile register file
// across TWO ASYNCHRONOUS DOMAINS (s_axi_aclk / core_clk).
//
// HOST CONTRACT: busy and done cross on independent synchronizer
// paths from the same core-domain event, so STATUS may transiently
// read busy=1,done=1 for ~1 aclk while a falling busy is still in
// flight. done is sticky and cleared only by the next go write -
// hosts poll done alone, never the combination (!busy && done).
//
// - After writing N_VECS or REQUANT_CFG, allow the crossing to
//   settle (~6 aclk + ~6 core clks; a rewrite while a handshake is
//   in flight defers delivery by another round trip) before writing
//   go. Back-to-back config-then-go can run with stale config.
// - A go written while busy is DROPPED by the core (accepted only
//   when idle) but still clears done here - so the current run's
//   completion would re-set done and look like the dropped command
//   finished. Poll done before re-arming go.
// - go/done pulse spacing relies on AXI transaction pacing and
//   assumes core_clk is not slower than ~aclk/3 (see pulse_sync).
// - wstrb is accepted but ignored: full-word 32-bit writes only.
// - When the core sits behind ws_tiling_engine (tiled top), go/busy/
//   done describe the OUTER sweep; K_TILES/N_TILES follow the same
//   settle-before-go rule as N_VECS. STATUS[2] (cfg_err, sync_2ff'd
//   level) means the last go was REJECTED by the engine (bad n_vecs/
//   tile counts/capacity): busy never rises and done never arrives -
//   but the go write already cleared done. A host whose done poll
//   times out must check STATUS[2] before assuming a hang.
//
// - POKE/PEEK are a DEBUG bring-up path, not the data path: one 32-bit
//   register write per lane. See the poke/peek contract below.
//
// Register map (byte offsets; word decode on addr[5:2]):
//   0x00 CTRL        W  [0] go - self-clearing pulse, clears done
//                       [1] poke_err clear (self-clearing pulse)
//   0x04 STATUS      R  [0] busy (sync_2ff), [1] done (sticky),
//                       [2] cfg_err (sync_2ff level),
//                       [3] poke_busy, [4] poke_err (sticky)
//   0x08 N_VECS      W  [VEC_W-1:0]; AXI-side readback supported
//   0x0C REQUANT_CFG W  [15:0] scale_mult, [20:16] shift_amt,
//                       [21] relu_en (22 bits - one register)
//   0x10 PERF_SEL    W  [3:0] counter index; readback supported
//   0x14 PERF_DATA   R  selected counter, refreshed once per PERF_SEL
//                       write (round trip ~4-6 aclk + ~6 core clks;
//                       hosts should budget 20 aclk - hosts write
//                       PERF_SEL, wait, then read)
//   0x18 K_TILES     W  [TILE_W-1:0]; AXI-side readback supported
//   0x1C N_TILES     W  [TILE_W-1:0]; AXI-side readback supported
//   0x20 POKE_ADDR   W  [5:0] word addr, [7:6] lane, [8] buf_sel
//                       (0=weight, 1=activation); readback supported
//   0x24 POKE_DATA   W  32-bit lane value; a write with lane ==
//                       BUF_WORD/32-1 COMMITS the assembled word
//   0x28 PEEK_ADDR   W  [5:0] idx0, [9:6] idx1, [11:10] buf_sel
//                       (0=weight, 1=activation, 2=res INT8,
//                       3=res INT32); readback supported
//                       buf_sel 2 returns REAL requantized INT8 from
//                       the capture banks (four packed columns per
//                       element, idx0=vector, idx1=lane). It read 0
//                       between a44910d and the requant pipelining -
//                       hosts written against that must switch back.
//   0x2C PEEK_DATA   R  selected 32-bit element; refreshed once per
//                       PEEK_ADDR write (same ~20 aclk budget as
//                       PERF_DATA - write address, wait, then read)
//
// POKE/PEEK HOST CONTRACT:
// - A buffer word is BUF_WORD bits (N*DATA_WIDTH = 128 at N=16) but
//   POKE_DATA is 32 bits, so a word takes BUF_WORD/32 lane writes.
//   Set POKE_ADDR (word + lane), write POKE_DATA, repeat; the write
//   to the last lane commits the whole word as ONE cross-domain
//   payload. Address and data are therefore atomic - there is no
//   settle-before-poke ordering hazard the way there is for N_VECS.
// - POLL STATUS[3] (poke_busy) BEFORE EACH COMMIT WRITE. bus_handshake
//   ignores a send while busy, so an unpaced commit is DROPPED and
//   that word is silently lost. A commit that lands on top of an
//   already-queued one sets STATUS[4] (poke_err, sticky, cleared by
//   CTRL[1]) so the loss is detectable on hardware instead of showing
//   up as a wrong answer. Re-delivering the same (addr,data) is
//   harmless - only drops matter.
// - Peek reads the REAL buffers, sharing the controller's read port
//   (see ws_unit's debug mux): the controller always wins, so a peek
//   issued while busy returns garbage but cannot disturb a run.
//   PEEK WHILE IDLE.
// - Poke covers weight/activation only. Result banks are chip-written;
//   peek reads them (buf_sel 2/3) but nothing can poke them.
//
// CDC per signal class (spec section 2): go/done pulse_sync, busy
// sync_2ff, every multi-bit value bus_handshake (req/ack, source
// holds value until acked - never parallel per-bit 2-flop chains).
// Dirty-flag relaunch: a register rewritten while its handshake is
// in flight is re-sent when the ack returns, so the core converges
// on the latest value and never observes a torn one.
//
// w_tile_idx/a_base have no registers (spec map); the single-tile
// integration (ws_tpu_top) ties them to 0, and ws_tpu_tiled_top's
// tiling engine drives them.
// Control plane only - zero DSPs expected in synth (a DSP here is a
// bug signal). Weight/activation data movement is NOT here (AXI-HP/
// DMA, separate work).
//====================================================================
module axi_lite_ctrl #(
    parameter int VEC_W       = 4,
    parameter int SCALE_WIDTH = 16,
    parameter int SHIFT_WIDTH = 5,
    parameter int TILE_W      = 4,
    parameter int BUF_DEPTH   = 64,   // tile_buffer DEPTH (words)
    parameter int BUF_WORD    = 32    // tile_buffer WORD_WIDTH = N*DATA_WIDTH
)(
    // AXI-Lite subordinate (PS domain)
    input  logic        s_axi_aclk,
    input  logic        s_axi_aresetn,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [5:0]  s_axi_awaddr,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    output logic [1:0]  s_axi_bresp,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    input  logic [5:0]  s_axi_araddr,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,

    // core domain
    input  logic                          core_clk,
    input  logic                          core_rst_n,
    output logic                          core_go,
    input  logic                          core_busy,
    output logic [VEC_W-1:0]              core_n_vecs,
    output logic [TILE_W-1:0]             core_k_tiles,
    output logic [TILE_W-1:0]             core_n_tiles,
    input  logic                          core_cfg_err,
    output logic                          core_relu_en,
    output logic signed [SCALE_WIDTH-1:0] core_scale_mult,
    output logic [SHIFT_WIDTH-1:0]        core_shift_amt,
    output logic [3:0]                    core_perf_addr,
    input  logic [31:0]                   core_perf_data,

    // debug poke: one-shot buffer write, already in the core domain.
    // core_poke_valid is a 1-cycle pulse - drive wr_en_* with it
    // directly, never a level.
    output logic                          core_poke_valid,
    output logic                          core_poke_sel,    // 0=weight 1=activation
    output logic [$clog2(BUF_DEPTH)-1:0]  core_poke_addr,
    output logic [BUF_WORD-1:0]           core_poke_data,

    // debug peek: address out (1-cycle valid pulse), 32-bit value back
    output logic [11:0]                   core_peek_addr,
    output logic                          core_peek_valid,
    input  logic [31:0]                   core_peek_data
);

    localparam int CFG_W = SCALE_WIDTH + SHIFT_WIDTH + 1;   // 22

    // Debug poke/peek geometry. One buffer word takes BUF_WORD/32
    // AXI writes; the write to the LAST lane commits the whole word.
    localparam int BUF_AW    = $clog2(BUF_DEPTH);
    localparam int NUM_LANES = BUF_WORD / 32;
    localparam int LANE_MAX  = NUM_LANES - 1;
    localparam int POKE_PW   = 1 + BUF_AW + BUF_WORD;   // {sel, addr, word}

    if (BUF_WORD % 32 != 0 || BUF_WORD > 128) begin : gen_poke_word_guard
        initial begin
            $error("axi_lite_ctrl: BUF_WORD (%0d) must be a multiple of 32 and <= 128 (POKE_ADDR has a 2-bit lane field)",
                   BUF_WORD);
            $fatal(1);
        end
    end

    if (BUF_AW > 6) begin : gen_poke_addr_guard
        initial begin
            $error("axi_lite_ctrl: BUF_DEPTH (%0d) needs %0d address bits; POKE_ADDR/PEEK_ADDR carry 6",
                   BUF_DEPTH, BUF_AW);
            $fatal(1);
        end
    end

    // word address = byte address [5:2]
    localparam logic [3:0] A_CTRL   = 4'h0;
    localparam logic [3:0] A_STATUS = 4'h1;
    localparam logic [3:0] A_NVECS  = 4'h2;
    localparam logic [3:0] A_CFG    = 4'h3;
    localparam logic [3:0] A_PSEL   = 4'h4;
    localparam logic [3:0] A_PDATA  = 4'h5;
    localparam logic [3:0] A_KTILES = 4'h6;   // 0x18
    localparam logic [3:0] A_NTILES = 4'h7;   // 0x1C
    localparam logic [3:0] A_POKEA  = 4'h8;   // 0x20
    localparam logic [3:0] A_POKED  = 4'h9;   // 0x24
    localparam logic [3:0] A_PEEKA  = 4'hA;   // 0x28
    localparam logic [3:0] A_PEEKD  = 4'hB;   // 0x2C

    //----------------------------------------------------------------
    // Write channel: capture AW / W independently, commit together
    //----------------------------------------------------------------
    logic        aw_got, w_got;
    logic [3:0]  waddr_q;
    logic [31:0] wdata_q;
    logic        commit;
    logic        go_write;

    assign s_axi_awready = !aw_got && !s_axi_bvalid;
    assign s_axi_wready  = !w_got  && !s_axi_bvalid;
    assign s_axi_bresp   = 2'b00;
    assign commit        = aw_got && w_got && !s_axi_bvalid;
    assign go_write      = commit && (waddr_q == A_CTRL) && wdata_q[0];

    // AXI-side register file + handshake launch state
    logic [VEC_W-1:0] nvecs_reg;
    logic [CFG_W-1:0] cfg_reg;
    logic [3:0]       psel_reg;
    logic [TILE_W-1:0] ktiles_reg;
    logic [TILE_W-1:0] ntiles_reg;
    logic nvecs_dirty, cfg_dirty, psel_dirty, ktiles_dirty, ntiles_dirty;
    logic nvecs_send,  cfg_send,  psel_send,  ktiles_send,  ntiles_send;
    logic nvecs_hs_busy, cfg_hs_busy, psel_hs_busy, ktiles_hs_busy, ntiles_hs_busy;
    logic go_pulse_axi;

    // debug poke/peek, AXI side. poke_stage is declared at the full
    // 128-bit width regardless of BUF_WORD so the lane part-select is
    // always in range; only [BUF_WORD-1:0] is ever sent.
    logic [8:0]   pokea_reg;      // [8] sel, [7:6] lane, [5:0] word addr
    logic [127:0] poke_stage;
    logic         poke_dirty, poke_send, poke_hs_busy, poke_err;
    logic [11:0]  peeka_reg;      // [11:10] buf_sel, [9:6] idx1, [5:0] idx0
    logic         peeka_dirty, peeka_send, peeka_hs_busy;
    logic         pokeerr_clr;
    logic [1:0]   poke_lane;

    assign poke_lane   = pokea_reg[7:6];
    assign pokeerr_clr = commit && (waddr_q == A_CTRL) && wdata_q[1];

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            aw_got       <= 1'b0;
            w_got        <= 1'b0;
            s_axi_bvalid <= 1'b0;
            waddr_q      <= '0;
            wdata_q      <= '0;
            nvecs_reg    <= '0;
            cfg_reg      <= '0;
            psel_reg     <= '0;
            ktiles_reg   <= '0;
            ntiles_reg   <= '0;
            nvecs_dirty  <= 1'b0;
            cfg_dirty    <= 1'b0;
            psel_dirty   <= 1'b0;
            ktiles_dirty <= 1'b0;
            ntiles_dirty <= 1'b0;
            nvecs_send   <= 1'b0;
            cfg_send     <= 1'b0;
            psel_send    <= 1'b0;
            ktiles_send  <= 1'b0;
            ntiles_send  <= 1'b0;
            go_pulse_axi <= 1'b0;
            pokea_reg    <= '0;
            poke_stage   <= '0;
            poke_dirty   <= 1'b0;
            poke_send    <= 1'b0;
            poke_err     <= 1'b0;
            peeka_reg    <= '0;
            peeka_dirty  <= 1'b0;
            peeka_send   <= 1'b0;
        end else begin
            nvecs_send   <= 1'b0;
            cfg_send     <= 1'b0;
            psel_send    <= 1'b0;
            ktiles_send  <= 1'b0;
            ntiles_send  <= 1'b0;
            poke_send    <= 1'b0;
            peeka_send   <= 1'b0;
            go_pulse_axi <= go_write;
            if (pokeerr_clr) poke_err <= 1'b0;

            if (s_axi_awvalid && s_axi_awready) begin
                waddr_q <= s_axi_awaddr[5:2];
                aw_got  <= 1'b1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                wdata_q <= s_axi_wdata;
                w_got   <= 1'b1;
            end

            // handshake launches: fire one cycle after the commit that
            // set dirty (holding registers are stable when sampled);
            // rewrites during flight relaunch after the ack returns.
            // Ordered BEFORE the commit block below so that, on a
            // same-edge coincidence of a launch and a new commit to
            // the same register, the commit's dirty-set (last
            // nonblocking assignment in program order) wins over this
            // block's dirty-clear and the new value relaunches after
            // the in-flight ack returns.
            if (nvecs_dirty && !nvecs_hs_busy && !nvecs_send) begin
                nvecs_send  <= 1'b1;
                nvecs_dirty <= 1'b0;
            end
            if (cfg_dirty && !cfg_hs_busy && !cfg_send) begin
                cfg_send  <= 1'b1;
                cfg_dirty <= 1'b0;
            end
            if (psel_dirty && !psel_hs_busy && !psel_send) begin
                psel_send  <= 1'b1;
                psel_dirty <= 1'b0;
            end
            if (ktiles_dirty && !ktiles_hs_busy && !ktiles_send) begin
                ktiles_send  <= 1'b1;
                ktiles_dirty <= 1'b0;
            end
            if (ntiles_dirty && !ntiles_hs_busy && !ntiles_send) begin
                ntiles_send  <= 1'b1;
                ntiles_dirty <= 1'b0;
            end
            if (poke_dirty && !poke_hs_busy && !poke_send) begin
                poke_send  <= 1'b1;
                poke_dirty <= 1'b0;
            end
            if (peeka_dirty && !peeka_hs_busy && !peeka_send) begin
                peeka_send  <= 1'b1;
                peeka_dirty <= 1'b0;
            end

            if (commit) begin
                case (waddr_q)
                    A_NVECS: begin
                        nvecs_reg   <= wdata_q[VEC_W-1:0];
                        nvecs_dirty <= 1'b1;
                    end
                    A_CFG: begin
                        cfg_reg   <= wdata_q[CFG_W-1:0];
                        cfg_dirty <= 1'b1;
                    end
                    A_PSEL: begin
                        psel_reg   <= wdata_q[3:0];
                        psel_dirty <= 1'b1;
                    end
                    A_KTILES: begin
                        ktiles_reg   <= wdata_q[TILE_W-1:0];
                        ktiles_dirty <= 1'b1;
                    end
                    A_NTILES: begin
                        ntiles_reg   <= wdata_q[TILE_W-1:0];
                        ntiles_dirty <= 1'b1;
                    end
                    A_POKEA: pokea_reg <= wdata_q[8:0];
                    A_POKED: begin
                        // deposit this lane; the LAST lane commits the
                        // assembled word as one atomic {sel,addr,data}
                        // payload. A commit landing while a previous one
                        // is still queued would overwrite poke_stage and
                        // silently lose that word - flag it instead.
                        poke_stage[poke_lane*32 +: 32] <= wdata_q;
                        if (poke_lane == LANE_MAX[1:0]) begin
                            poke_dirty <= 1'b1;
                            if (poke_dirty) poke_err <= 1'b1;
                        end
                    end
                    A_PEEKA: begin
                        peeka_reg   <= wdata_q[11:0];
                        peeka_dirty <= 1'b1;
                    end
                    default: ;      // CTRL handled via go_write; others ignored
                endcase
                s_axi_bvalid <= 1'b1;
                aw_got       <= 1'b0;
                w_got        <= 1'b0;
            end
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;
        end
    end

    //----------------------------------------------------------------
    // CDC: go pulse into the core domain
    //----------------------------------------------------------------
    pulse_sync u_cdc_go (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_pulse (go_pulse_axi),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_pulse (core_go)
    );

    //----------------------------------------------------------------
    // CDC: busy level into the AXI domain
    //----------------------------------------------------------------
    logic busy_axi;

    sync_2ff u_cdc_busy (
        .clk   (s_axi_aclk),
        .rst_n (s_axi_aresetn),
        .d     (core_busy),
        .q     (busy_axi)
    );

    logic cfgerr_axi;

    sync_2ff u_cdc_cfgerr (
        .clk   (s_axi_aclk),
        .rst_n (s_axi_aresetn),
        .d     (core_cfg_err),
        .q     (cfgerr_axi)
    );

    //----------------------------------------------------------------
    // done: busy falling edge detected in the CORE domain (exact),
    // crossed as a pulse, sticky in the AXI domain until a go write
    //----------------------------------------------------------------
    logic core_busy_q;
    logic core_done_pulse;
    logic done_pulse_axi;
    logic done_ff;

    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n)
            core_busy_q <= 1'b0;
        else
            core_busy_q <= core_busy;
    end

    assign core_done_pulse = core_busy_q && !core_busy;

    pulse_sync u_cdc_done (
        .src_clk   (core_clk),
        .src_rst_n (core_rst_n),
        .src_pulse (core_done_pulse),
        .dst_clk   (s_axi_aclk),
        .dst_rst_n (s_axi_aresetn),
        .dst_pulse (done_pulse_axi)
    );

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn)
            done_ff <= 1'b0;
        else if (go_write)
            done_ff <= 1'b0;
        else if (done_pulse_axi)
            done_ff <= 1'b1;
    end

    //----------------------------------------------------------------
    // CDC: multi-bit values, AXI -> core (never per-bit sync chains)
    //----------------------------------------------------------------
    logic [VEC_W-1:0] nvecs_core;
    logic [CFG_W-1:0] cfg_core;
    logic nvecs_valid_core, cfg_valid_core, psel_valid_core;

    bus_handshake #(.W(VEC_W)) u_hs_nvecs (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_data  (nvecs_reg),
        .src_send  (nvecs_send),
        .src_busy  (nvecs_hs_busy),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_data  (nvecs_core),
        .dst_valid (nvecs_valid_core)
    );

    assign core_n_vecs = nvecs_core;

    bus_handshake #(.W(CFG_W)) u_hs_cfg (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_data  (cfg_reg),
        .src_send  (cfg_send),
        .src_busy  (cfg_hs_busy),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_data  (cfg_core),
        .dst_valid (cfg_valid_core)
    );

    assign core_scale_mult = cfg_core[SCALE_WIDTH-1:0];
    assign core_shift_amt  = cfg_core[SCALE_WIDTH+SHIFT_WIDTH-1:SCALE_WIDTH];
    assign core_relu_en    = cfg_core[CFG_W-1];

    bus_handshake #(.W(4)) u_hs_psel (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_data  (psel_reg),
        .src_send  (psel_send),
        .src_busy  (psel_hs_busy),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_data  (core_perf_addr),
        .dst_valid (psel_valid_core)
    );

    logic ktiles_valid_core, ntiles_valid_core;

    bus_handshake #(.W(TILE_W)) u_hs_ktiles (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_data  (ktiles_reg),
        .src_send  (ktiles_send),
        .src_busy  (ktiles_hs_busy),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_data  (core_k_tiles),
        .dst_valid (ktiles_valid_core)
    );

    bus_handshake #(.W(TILE_W)) u_hs_ntiles (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_data  (ntiles_reg),
        .src_send  (ntiles_send),
        .src_busy  (ntiles_hs_busy),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_data  (core_n_tiles),
        .dst_valid (ntiles_valid_core)
    );

    //----------------------------------------------------------------
    // Debug poke: address AND data cross as ONE payload, so the pair
    // is atomic - a separately-crossed address could be stale when the
    // data lands and would silently write the wrong location. The
    // handshake's dst_valid is already a 1-cycle pulse per delivery,
    // so it drives the buffer write enable directly (exactly once per
    // host commit) with no level-sensitive gating.
    //----------------------------------------------------------------
    logic [POKE_PW-1:0] poke_payload;
    logic [POKE_PW-1:0] poke_core;

    assign poke_payload = {pokea_reg[8], pokea_reg[BUF_AW-1:0], poke_stage[BUF_WORD-1:0]};

    bus_handshake #(.W(POKE_PW)) u_hs_poke (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_data  (poke_payload),
        .src_send  (poke_send),
        .src_busy  (poke_hs_busy),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_data  (poke_core),
        .dst_valid (core_poke_valid)
    );

    assign core_poke_data = poke_core[BUF_WORD-1:0];
    assign core_poke_addr = poke_core[BUF_WORD+BUF_AW-1:BUF_WORD];
    assign core_poke_sel  = poke_core[POKE_PW-1];

    //----------------------------------------------------------------
    // Debug peek: same round trip shape as PERF_SEL/PERF_DATA - the
    // address crosses over, the core does a REGISTERED read, the
    // 32-bit result crosses back. Host contract matches PERF exactly:
    // write PEEK_ADDR, wait ~20 aclk, read PEEK_DATA.
    //----------------------------------------------------------------
    bus_handshake #(.W(12)) u_hs_peeka (
        .src_clk   (s_axi_aclk),
        .src_rst_n (s_axi_aresetn),
        .src_data  (peeka_reg),
        .src_send  (peeka_send),
        .src_busy  (peeka_hs_busy),
        .dst_clk   (core_clk),
        .dst_rst_n (core_rst_n),
        .dst_data  (core_peek_addr),
        .dst_valid (core_peek_valid)
    );

    // three cycles of settle: the peek read is registered on both
    // sides (address reg -> memory/bank read -> output reg), so the
    // value is stable by d3. One more than PERF needs, for the extra
    // pipeline stage that keeps the wide read mux off the timing path.
    logic peek_v_d1, peek_v_d2, peek_v_d3;
    logic peekd_dirty, peekd_send, peekd_hs_busy;
    logic [31:0] peekd_shadow;
    logic peekd_valid_axi;

    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            peek_v_d1   <= 1'b0;
            peek_v_d2   <= 1'b0;
            peek_v_d3   <= 1'b0;
            peekd_dirty <= 1'b0;
            peekd_send  <= 1'b0;
        end else begin
            peek_v_d1  <= core_peek_valid;
            peek_v_d2  <= peek_v_d1;
            peek_v_d3  <= peek_v_d2;
            peekd_send <= 1'b0;
            if (peek_v_d3)
                peekd_dirty <= 1'b1;
            else if (peekd_dirty && !peekd_hs_busy && !peekd_send) begin
                peekd_send  <= 1'b1;
                peekd_dirty <= 1'b0;
            end
        end
    end

    bus_handshake #(.W(32)) u_hs_peekd (
        .src_clk   (core_clk),
        .src_rst_n (core_rst_n),
        .src_data  (core_peek_data),
        .src_send  (peekd_send),
        .src_busy  (peekd_hs_busy),
        .dst_clk   (s_axi_aclk),
        .dst_rst_n (s_axi_aresetn),
        .dst_data  (peekd_shadow),
        .dst_valid (peekd_valid_axi)
    );

    //----------------------------------------------------------------
    // Perf response: core_perf_data is valid one core cycle after
    // core_perf_addr lands (registered read port); wait two cycles
    // for margin, then hand the value back across. Same dirty-flag
    // relaunch so a hammered PERF_SEL still delivers a final answer.
    //----------------------------------------------------------------
    logic psel_v_d1, psel_v_d2;
    logic pdata_dirty, pdata_send;
    logic pdata_hs_busy;
    logic [31:0] pdata_shadow;
    logic pdata_valid_axi;

    always_ff @(posedge core_clk or negedge core_rst_n) begin
        if (!core_rst_n) begin
            psel_v_d1   <= 1'b0;
            psel_v_d2   <= 1'b0;
            pdata_dirty <= 1'b0;
            pdata_send  <= 1'b0;
        end else begin
            psel_v_d1  <= psel_valid_core;
            psel_v_d2  <= psel_v_d1;
            pdata_send <= 1'b0;
            if (psel_v_d2)
                pdata_dirty <= 1'b1;
            else if (pdata_dirty && !pdata_hs_busy && !pdata_send) begin
                pdata_send  <= 1'b1;
                pdata_dirty <= 1'b0;
            end
        end
    end

    bus_handshake #(.W(32)) u_hs_pdata (
        .src_clk   (core_clk),
        .src_rst_n (core_rst_n),
        .src_data  (core_perf_data),
        .src_send  (pdata_send),
        .src_busy  (pdata_hs_busy),
        .dst_clk   (s_axi_aclk),
        .dst_rst_n (s_axi_aresetn),
        .dst_data  (pdata_shadow),
        .dst_valid (pdata_valid_axi)
    );

    //----------------------------------------------------------------
    // Read channel: single outstanding, registered data
    //----------------------------------------------------------------
    assign s_axi_arready = !s_axi_rvalid;
    assign s_axi_rresp   = 2'b00;

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata  <= '0;
        end else begin
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                case (s_axi_araddr[5:2])
                    A_STATUS: s_axi_rdata <= {27'b0, poke_err, poke_hs_busy,
                                              cfgerr_axi, done_ff, busy_axi};
                    A_NVECS:  s_axi_rdata <= {{(32-VEC_W){1'b0}}, nvecs_reg};
                    A_CFG:    s_axi_rdata <= {{(32-CFG_W){1'b0}}, cfg_reg};
                    A_PSEL:   s_axi_rdata <= {28'b0, psel_reg};
                    A_PDATA:  s_axi_rdata <= pdata_shadow;
                    A_KTILES: s_axi_rdata <= {{(32-TILE_W){1'b0}}, ktiles_reg};
                    A_NTILES: s_axi_rdata <= {{(32-TILE_W){1'b0}}, ntiles_reg};
                    A_POKEA:  s_axi_rdata <= {23'b0, pokea_reg};
                    A_PEEKA:  s_axi_rdata <= {20'b0, peeka_reg};
                    A_PEEKD:  s_axi_rdata <= peekd_shadow;
                    default:  s_axi_rdata <= '0;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
