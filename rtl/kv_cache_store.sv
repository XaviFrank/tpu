`timescale 1ns / 1ps
//====================================================================
// kv_cache_store: append-only quantized KV cache, on-chip resident.
// Step 4 of the KV cache write path.
//
//====================================================================
// SCOPE - on-chip is the test article, not the destination
//   At INT8 the full 40960-token context is ~2.35 GB, so DRAM
//   residency is unavoidable at real context lengths REGARDLESS of how
//   well this compresses. 3 bits/channel takes that to ~880 MB, which
//   is still far past any on-chip budget. So this module is not the
//   eventual cache; it is the amount of cache needed to prove the
//   quantize/append mechanism correct, and DRAM-backed addressing is
//   real deferred work rather than a detail.
//
//   MAX_POS is therefore deliberately small and parameterized. What
//   must be right NOW is the addressing arithmetic and the behaviour
//   at the capacity boundary, both of which survive the move to DRAM.
//
//====================================================================
// TIGHT STRIDE, AND WHY IT IS WORTH A MULTIPLIER
//   WORDS_PER_VEC is 13 or 14, neither a power of two. Padding each
//   vector to 16 words would make the stride a shift instead of a
//   multiply - and would waste 2 or 3 words per vector, 14% to 19% of
//   the cache. In a structure whose entire purpose is capacity, paying
//   a small constant multiplier to keep the packing tight is obviously
//   the right side of that trade. The multiply is by a CONSTANT and
//   only on the read port; synthesis reduces it to shifts and adds
//   (14 = 8+4+2), so it costs no DSP.
//
//   The write port has no multiply at all: the base pointer advances
//   by WORDS_PER_VEC on commit, so writes are base+offset.
//
//====================================================================
// OVERFLOW IS EXPLICIT, NOT A WRAP
//   When the cache is full, further commits set a sticky overflow flag
//   and writes are suppressed. A silent wrap would corrupt the oldest
//   entries while every interface still looked healthy - the worst
//   available failure mode for a structure the attention read side
//   will trust. Making the boundary loud is the point; the testbench
//   fills the cache, attempts one more append, and checks BOTH that
//   overflow asserts AND that the last good entry is untouched.
//====================================================================
module kv_cache_store #(
    parameter int WORDS_PER_VEC = 14,
    parameter int MAX_POS       = 64
)(
    input  logic clk,
    input  logic rst_n,
    input  logic clear,

    // write side, driven straight from kv_quantize
    input  logic        w_valid,
    input  logic [4:0]  w_idx,
    input  logic [31:0] w_data,
    input  logic        commit,

    output logic [$clog2(MAX_POS+1)-1:0] n_pos,
    output logic        full,
    output logic        overflow,

    // read side, for verification and for the deferred attention path
    input  logic        rd_en,
    input  logic [$clog2(MAX_POS+1)-1:0] rd_pos,
    input  logic [4:0]  rd_widx,
    output logic [31:0] rd_data
);

    localparam int POSW  = $clog2(MAX_POS+1);
    localparam int DEPTH = MAX_POS * WORDS_PER_VEC;
    localparam int AW    = $clog2(DEPTH);

    if (DEPTH > 65536) begin : gen_depth_max
        initial begin
            $error("kv_cache_store: DEPTH (%0d) exceeds the 16-bit address math",
                   DEPTH);
            $fatal(1);
        end
    end

    (* ram_style = "block" *) logic [31:0] mem [DEPTH];

    // Address arithmetic is done at a FIXED 16 bits and then narrowed,
    // rather than at AW. Deriving the widths from AW looks tidier but
    // breaks for small MAX_POS: a 5-bit word offset zero-extended to
    // AW is a zero-width replication once AW <= 5, which is exactly
    // the configuration a round-trip testbench wants to instantiate.
    // The upper bits fold away in synthesis, so this costs nothing.
    logic [15:0]     wbase, waddr, raddr, rmul;
    logic [POSW-1:0] pos;

    assign full  = (pos == MAX_POS[POSW-1:0]);
    assign n_pos = pos;

    assign waddr = wbase + {11'b0, w_idx};
    assign rmul  = {{(16-POSW){1'b0}}, rd_pos} * WORDS_PER_VEC[15:0];
    assign raddr = rmul + {11'b0, rd_widx};

    // Datapath memory: no reset, per house style.
    always_ff @(posedge clk) begin
        if (w_valid && !full) mem[waddr[AW-1:0]] <= w_data;
    end

    always_ff @(posedge clk) begin
        if (rd_en) rd_data <= mem[raddr[AW-1:0]];
    end

    // Control registers: async reset.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pos      <= '0;
            wbase    <= '0;
            overflow <= 1'b0;
        end else if (clear) begin
            pos      <= '0;
            wbase    <= '0;
            overflow <= 1'b0;
        end else if (commit) begin
            if (full) begin
                overflow <= 1'b1;
            end else begin
                pos   <= pos + 1'b1;
                wbase <= wbase + WORDS_PER_VEC[15:0];
            end
        end
    end

endmodule
