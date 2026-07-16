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
// Register map (byte offsets; word decode on addr[5:2]):
//   0x00 CTRL        W  [0] go - self-clearing pulse, clears done
//   0x04 STATUS      R  [0] busy (sync_2ff), [1] done (sticky)
//   0x08 N_VECS      W  [VEC_W-1:0]; AXI-side readback supported
//   0x0C REQUANT_CFG W  [15:0] scale_mult, [20:16] shift_amt,
//                       [21] relu_en (22 bits - one register)
//   0x10 PERF_SEL    W  [3:0] counter index; readback supported
//   0x14 PERF_DATA   R  selected counter, refreshed once per PERF_SEL
//                       write (bounded ~4 aclk + ~6 core clks later -
//                       hosts write PERF_SEL, wait, then read)
//
// CDC per signal class (spec section 2): go/done pulse_sync, busy
// sync_2ff, every multi-bit value bus_handshake (req/ack, source
// holds value until acked - never parallel per-bit 2-flop chains).
// Dirty-flag relaunch: a register rewritten while its handshake is
// in flight is re-sent when the ack returns, so the core converges
// on the latest value and never observes a torn one.
//
// Single-tile scope: w_tile_idx/a_base have no registers (spec map);
// the integration ties them to 0. First extension for WS tiling.
// Control plane only - zero DSPs expected in synth (a DSP here is a
// bug signal). Weight/activation data movement is NOT here (AXI-HP/
// DMA, separate work).
//====================================================================
module axi_lite_ctrl #(
    parameter int VEC_W       = 4,
    parameter int SCALE_WIDTH = 16,
    parameter int SHIFT_WIDTH = 5
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
    output logic                          core_relu_en,
    output logic signed [SCALE_WIDTH-1:0] core_scale_mult,
    output logic [SHIFT_WIDTH-1:0]        core_shift_amt,
    output logic [3:0]                    core_perf_addr,
    input  logic [31:0]                   core_perf_data
);

    localparam int CFG_W = SCALE_WIDTH + SHIFT_WIDTH + 1;   // 22

    // word address = byte address [5:2]
    localparam logic [3:0] A_CTRL   = 4'h0;
    localparam logic [3:0] A_STATUS = 4'h1;
    localparam logic [3:0] A_NVECS  = 4'h2;
    localparam logic [3:0] A_CFG    = 4'h3;
    localparam logic [3:0] A_PSEL   = 4'h4;
    localparam logic [3:0] A_PDATA  = 4'h5;

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
    logic nvecs_dirty, cfg_dirty, psel_dirty;
    logic nvecs_send,  cfg_send,  psel_send;
    logic nvecs_hs_busy, cfg_hs_busy, psel_hs_busy;
    logic go_pulse_axi;

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
            nvecs_dirty  <= 1'b0;
            cfg_dirty    <= 1'b0;
            psel_dirty   <= 1'b0;
            nvecs_send   <= 1'b0;
            cfg_send     <= 1'b0;
            psel_send    <= 1'b0;
            go_pulse_axi <= 1'b0;
        end else begin
            nvecs_send   <= 1'b0;
            cfg_send     <= 1'b0;
            psel_send    <= 1'b0;
            go_pulse_axi <= go_write;

            if (s_axi_awvalid && s_axi_awready) begin
                waddr_q <= s_axi_awaddr[5:2];
                aw_got  <= 1'b1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                wdata_q <= s_axi_wdata;
                w_got   <= 1'b1;
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
                    default: ;      // CTRL handled via go_write; others ignored
                endcase
                s_axi_bvalid <= 1'b1;
                aw_got       <= 1'b0;
                w_got        <= 1'b0;
            end
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;

            // handshake launches: fire one cycle after the commit that
            // set dirty (holding registers are stable when sampled);
            // rewrites during flight relaunch after the ack returns
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
                    A_STATUS: s_axi_rdata <= {30'b0, done_ff, busy_axi};
                    A_NVECS:  s_axi_rdata <= {{(32-VEC_W){1'b0}}, nvecs_reg};
                    A_CFG:    s_axi_rdata <= {{(32-CFG_W){1'b0}}, cfg_reg};
                    A_PSEL:   s_axi_rdata <= {28'b0, psel_reg};
                    A_PDATA:  s_axi_rdata <= pdata_shadow;
                    default:  s_axi_rdata <= '0;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule
