`timescale 1ns / 1ps
//====================================================================
// axi_stream_to_buf_adapter: AXI4-Stream -> raw buffer write port.
//
//   axi_dma's M_AXIS_MM2S speaks AXI4-Stream (tdata/tvalid/tready/
//   tlast); nothing in this design does. This is the thin shim that
//   turns accepted stream beats into writes on one of the existing
//   raw buffer write ports (wr_en/wr_addr/wr_data), which until now
//   had no driver except poke's one-word-at-a-time register writes.
//
//   TARGET-AGNOSTIC BY DESIGN. There is deliberately no weight-vs-
//   activation parameter: the two differ only in which port the
//   instance is wired to, so this is one module type with two thin
//   instantiations, the same pattern requant.sv uses across its call
//   sites. Nothing here knows or cares which buffer it feeds.
//
//====================================================================
// DELIBERATELY DUMB v1 - WHAT THIS DOES NOT DO
//====================================================================
//   This is the simplest thing that can correctly land bytes, and
//   that is a choice, not an oversight. Anything smarter would mean
//   guessing at requirements that depend on data this project does
//   not have yet: measured DDR bandwidth, and the real batch/tiling
//   sweep order under load.
//
//   NOT here, all deferred deliberately:
//   - No tile- or stride-aware addressing. This adapter has ZERO
//     knowledge of ws_tiling_engine's sweep order (N-outer/K-inner)
//     or of w_tile_idx / a_base. It counts 0,1,2,... and nothing
//     else. Real integration with the tiling engine's access pattern
//     is a known, deferred enhancement.
//   - No scatter-gather, no multi-buffer interleaving within one
//     transfer. One stream feeds one buffer, sequentially.
//   - No coupling to bandwidth or batch-size decisions. It does not
//     know n_vecs and does not care; it moves whatever beats it is
//     given, in order.
//
//   FUTURE WORK: tighten the addressing once real DDR bandwidth is
//   measured and the tiling engine's real sweep order needs to drive
//   where data lands, rather than dumb sequential counting. Expect
//   that to add tile/stride inputs here, or to replace this module
//   with a descriptor-driven one.
//
//====================================================================
// INTEGRATION CONTRACT - read before wiring this in
//====================================================================
//   - ONE BEAT = ONE BUFFER WORD. tdata is WORD_W wide and lands
//     whole; there is no packing or unpacking. The DMA's stream data
//     width MUST be configured to WORD_W (N*DATA_WIDTH = 128 bits at
//     N=16) or beats will not map 1:1 onto buffer words. A narrower
//     DMA stream needs a packer that this version does not provide.
//   - buf_ready IS THE BACKPRESSURE SOURCE, and the integrator must
//     supply it. tile_buffer's write port has no ready of its own -
//     it accepts a write every cycle unconditionally - but its
//     contract is that a tile is never read while being written.
//
//     RESOLVED: drive it from !ws_tpu_top.core_busy_out. That port
//     was added for exactly this and is taken in the CORE domain,
//     before STATUS[0]'s synchronizer, because this adapter runs on
//     the same clock (pl_clk1) - a synchronizer here would make the
//     backpressure late, which is the window being closed. The
//     original block design tied buf_ready to an xlconstant(1);
//     tb_dma_weight_load is the simulation of the corrected wiring.
//
//     WHAT !busy DOES AND DOES NOT BUY. It stops beats landing
//     underneath a run that is ALREADY going. It does NOT stop a run
//     from STARTING mid-transfer - go is a host write, and if the
//     host issues it while beats are still in flight the core reads a
//     half-written tile and returns a confidently wrong answer. That
//     direction stays a HOST contract (wait for the DMA's idle bit
//     before writing go), exactly as poke's contract is to poll
//     poke_busy before go. Closing it in hardware needs a
//     transfer-in-flight signal gating go, which is a controller
//     change and deliberately not here.
//   - NO BEAT IS DROPPED under backpressure, by construction: a beat
//     is consumed only when tvalid && tready, and wr_en asserts only
//     on that same condition. tready is independent of tvalid (legal
//     AXI4-Stream) so there is no handshake deadlock.
//   - PRIORITY vs POKE: ws_tpu_top's poke mux gives the external
//     wr_* ports priority (mux_wr_addr = wr_en ? wr_addr : poke_addr).
//     So while this adapter holds wr_en high, poke writes to that
//     buffer are locked out. DMA and poke are alternatives, not
//     concurrent writers - do not interleave them.
//   - Pure control/addressing logic: ZERO DSPs expected. A DSP here
//     is a bug signal.
//====================================================================
module axi_stream_to_buf_adapter #(
    parameter int WORD_W = 32,   // one stream beat = one buffer word
    parameter int DEPTH  = 64    // target buffer depth, in words
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // AXI4-Stream slave (from axi_dma M_AXIS_MM2S)
    input  logic                    s_tvalid,
    output logic                    s_tready,
    input  logic [WORD_W-1:0]       s_tdata,
    input  logic                    s_tlast,

    // downstream readiness - integrator-supplied, see the contract
    input  logic                    buf_ready,

    // raw buffer write port (drives wr_en_*/wr_addr_*/wr_data_*)
    output logic                    wr_en,
    output logic [$clog2(DEPTH)-1:0] wr_addr,
    output logic [WORD_W-1:0]       wr_data,

    // sticky: the address counter ran past DEPTH-1 and wrapped, so an
    // oversized transfer has overwritten earlier words. Cleared only
    // by reset. Nothing consumes this yet (the adapter is not inside
    // ws_tpu_top) - it exists so an oversized transfer is DETECTABLE
    // rather than silently corrupting, the same reasoning as
    // axi_lite_ctrl's poke_err.
    output logic                    wrapped
);

    localparam int AW = $clog2(DEPTH);
    localparam logic [AW-1:0] ADDR_MAX = DEPTH - 1;

    if (DEPTH < 2) begin : gen_depth_guard
        initial begin
            $error("axi_stream_to_buf_adapter: DEPTH (%0d) must be >= 2", DEPTH);
            $fatal(1);
        end
    end

    logic [AW-1:0] addr_q;
    logic          accept;

    // tready does NOT depend on tvalid, which keeps the handshake
    // simple and is explicitly allowed by AXI4-Stream.
    assign s_tready = buf_ready;
    assign accept   = s_tvalid && s_tready;

    // Straight passthrough: the buffer latches on the same edge the
    // beat is accepted, so no holding register is needed and no beat
    // can be lost between the two.
    assign wr_en   = accept;
    assign wr_addr = addr_q;
    assign wr_data = s_tdata;

    // Address counter (control: async reset). tlast ends a transfer,
    // so the next one starts at 0; a transfer longer than the buffer
    // wraps and flags it. No tile-boundary logic on top of this.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            addr_q  <= '0;
            wrapped <= 1'b0;
        end else if (accept) begin
            if (s_tlast)
                addr_q <= '0;
            else if (addr_q == ADDR_MAX) begin
                addr_q  <= '0;
                wrapped <= 1'b1;
            end else
                addr_q <= addr_q + 1'b1;
        end
    end

endmodule
