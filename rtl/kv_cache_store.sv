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
//   Both ports carry that multiply now. The write port used to advance
//   a running base pointer instead, which is cheaper by one multiply -
//   but with layers the pointer has to be per-layer AND initialised to
//   L*LSTRIDE rather than zero, and an array of pointers is more state
//   and more ways to get out of step than a recomputed address. Every
//   multiply here is by a compile-time constant.
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
//
//   Same treatment for an out-of-range layer index, which is only
//   reachable when NUM_LAYERS is not a power of two (28 is not): the
//   commit is suppressed and overflow latches, rather than aliasing a
//   whole cache line onto some other layer's storage.
//
//====================================================================
// LAYER OFFSET - ONE CACHE, NUM_LAYERS REGIONS
//   A transformer layer's KV cache is private to that layer, so a
//   28-layer decode needs 28 of them. Instantiating 28 stores and
//   carving one store into 28 regions cost the SAME on-chip memory,
//   and neither reaches useful context length - 28 layers x 8 KV heads
//   x 40960 positions is ~1.65 GB, so DRAM residency is unavoidable
//   either way (see the scope note above).
//
//   The tie-breaker is therefore which one survives the move to DRAM,
//   and it is this one. A layer index folded into an address is the
//   same mechanism a DRAM-backed cache needs - one address space, one
//   read port, a base per region - so the addressing built here is
//   reused rather than discarded. 28 separate instances would have to
//   be collapsed back into one address space to talk to a memory
//   controller at all.
//
//   WHAT NUM_LAYERS=28 ACTUALLY COSTS, so MAX_POS is not set by hope.
//   One (layer, KV-head) pair costs 45 words per position - 32 for K,
//   13 for V - i.e. 180 B. The full model is 28 layers x 8 KV heads =
//   224 pairs, so ONE position of context is 40.3 KB across all the
//   stores. Against a few Mbit of on-chip BRAM that is roughly a dozen
//   positions of full-model context. The layer offset makes 28 layers
//   ADDRESSABLE; it does not make them fit, and nothing on-chip will.
//   That is the point - it is the addressing a DRAM-backed cache will
//   reuse, arrived at while it is still cheap to get wrong.
//
//   Layout is region-major: layer L owns words
//   [L*MAX_POS*WORDS_PER_VEC, (L+1)*MAX_POS*WORDS_PER_VEC). Positions
//   within a layer stay contiguous, which is what the read side
//   streams, and a layer's region is contiguous, which is what a DRAM
//   burst wants. Interleaving layers per position would break both.
//
//   Each layer keeps its OWN position count. In decode every layer
//   appends exactly once per token so the counts move together, and a
//   single shared counter would work - by convention. Per-layer
//   counters cost NUM_LAYERS*POSW flops (196 at 28 layers) and make
//   the structure correct instead of correct-by-convention, which
//   matters during prefill and whenever a layer loop is interrupted.
//====================================================================
module kv_cache_store #(
    parameter int WORDS_PER_VEC = 14,
    parameter int MAX_POS       = 64,
    parameter int NUM_LAYERS    = 1,
    // DERIVED - do not override. It lives in the parameter list only
    // because a port width cannot reference a localparam, and
    // $clog2(1) is 0, which would be a zero-width port at the default.
    parameter int LAYW          = (NUM_LAYERS <= 1) ? 1
                                                    : $clog2(NUM_LAYERS)
)(
    input  logic clk,
    input  logic rst_n,

    // Empties EVERY layer. A new sequence invalidates all of them, so
    // a per-layer clear would be speculative; add it when something
    // needs to evict one layer alone.
    input  logic clear,

    // write side, driven straight from kv_quantize
    input  logic [LAYW-1:0] w_layer,
    input  logic        w_valid,
    input  logic [4:0]  w_idx,
    input  logic [31:0] w_data,
    input  logic        commit,

    // n_pos and full report the layer selected by w_layer
    output logic [$clog2(MAX_POS+1)-1:0] n_pos,
    output logic        full,
    output logic        overflow,

    // read side, for verification and for the deferred attention path
    input  logic        rd_en,
    input  logic [LAYW-1:0] rd_layer,
    input  logic [$clog2(MAX_POS+1)-1:0] rd_pos,
    input  logic [4:0]  rd_widx,
    output logic [31:0] rd_data
);

    localparam int POSW    = $clog2(MAX_POS+1);
    localparam int LSTRIDE = MAX_POS * WORDS_PER_VEC;
    localparam int DEPTH   = NUM_LAYERS * LSTRIDE;
    localparam int AW      = $clog2(DEPTH);

    localparam logic [LAYW:0] NL_C = NUM_LAYERS[LAYW:0];

    // The old ceiling was the 16-bit address math, which the layer
    // offset outgrew immediately (28 x 102 x 32 = 91392 words). The
    // math is 32-bit now, so the real ceiling is on-chip memory: 256 K
    // words is 1 MB, already past any part this targets, and anything
    // near it should be in DRAM rather than here.
    if (DEPTH > 262144) begin : gen_depth_max
        initial begin
            $error("kv_cache_store: DEPTH (%0d words) is past any on-chip budget",
                   DEPTH);
            $fatal(1);
        end
    end

    (* ram_style = "block" *) logic [31:0] mem [DEPTH];

    // Address arithmetic is done at a FIXED 32 bits and then narrowed,
    // rather than at AW. Deriving the widths from AW looks tidier but
    // breaks for small MAX_POS: a 5-bit word offset zero-extended to
    // AW is a zero-width replication once AW <= 5, which is exactly
    // the configuration a round-trip testbench wants to instantiate.
    // The upper bits fold away in synthesis, so this costs nothing.
    //
    // The write port now carries the same constant multiply the read
    // port always had - it used to advance a running base pointer
    // instead, which is one multiply cheaper but needs a per-layer
    // pointer initialised to L*LSTRIDE. Both multiplies are by
    // CONSTANTS (LSTRIDE, WORDS_PER_VEC), so synthesis reduces them to
    // shifts and adds and neither costs a DSP; paying the second one
    // buys a symmetric read/write path and one less array of state.
    logic [31:0]     wlbase, wmul, waddr;
    logic [31:0]     rlbase, rmul, raddr;
    logic [POSW-1:0] pos [NUM_LAYERS];
    logic [POSW-1:0] wpos;
    logic [LAYW-1:0] wl_i, rl_i;
    logic            w_bad, r_bad;

    // Out-of-range only when NUM_LAYERS is not a power of two. Clamp
    // the array index so a bad request reads real state rather than X,
    // and let the flags below make it loud.
    assign w_bad = ({1'b0, w_layer}  >= NL_C);
    assign r_bad = ({1'b0, rd_layer} >= NL_C);
    assign wl_i  = w_bad ? {LAYW{1'b0}} : w_layer;
    assign rl_i  = r_bad ? {LAYW{1'b0}} : rd_layer;

    assign wpos  = pos[wl_i];
    assign full  = (wpos == MAX_POS[POSW-1:0]);
    assign n_pos = wpos;

    assign wlbase = {{(32-LAYW){1'b0}}, wl_i}  * LSTRIDE[31:0];
    assign wmul   = {{(32-POSW){1'b0}}, wpos}  * WORDS_PER_VEC[31:0];
    assign waddr  = wlbase + wmul + {27'b0, w_idx};

    assign rlbase = {{(32-LAYW){1'b0}}, rl_i}  * LSTRIDE[31:0];
    assign rmul   = {{(32-POSW){1'b0}}, rd_pos} * WORDS_PER_VEC[31:0];
    assign raddr  = rlbase + rmul + {27'b0, rd_widx};

    // Datapath memory: no reset, per house style.
    always_ff @(posedge clk) begin
        if (w_valid && !full && !w_bad) mem[waddr[AW-1:0]] <= w_data;
    end

    always_ff @(posedge clk) begin
        if (rd_en) rd_data <= mem[raddr[AW-1:0]];
    end

    // Control registers: async reset. The per-layer counters are reset
    // by an unrolled loop rather than a whole-array assignment, which
    // Icarus does not accept.
    integer li;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (li = 0; li < NUM_LAYERS; li = li + 1) pos[li] <= '0;
            overflow <= 1'b0;
        end else if (clear) begin
            for (li = 0; li < NUM_LAYERS; li = li + 1) pos[li] <= '0;
            overflow <= 1'b0;
        end else if (commit) begin
            if (full || w_bad) begin
                overflow <= 1'b1;
            end else begin
                pos[wl_i] <= wpos + 1'b1;
            end
        end
    end

endmodule
