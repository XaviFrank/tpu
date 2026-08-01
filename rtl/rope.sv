`timescale 1ns / 1ps
//====================================================================
// rope: rotary position embedding - phase generation.
//
//   This commit is the PHASE/ANGLE half: per-frequency phase
//   accumulation and cos/sin generation. The rotation datapath that
//   consumes them lands next.
//
//====================================================================
// TABLE DECISION - incremental phase, not a position x frequency grid
//====================================================================
//   Sized against this model's real config (head_dim 128, base 1e6,
//   max_position_embeddings 40960), on a device with 5.2 Mbit of BRAM:
//
//     full table, 40960 positions   83.9 Mbit   16x the WHOLE device
//     full table,  4096 positions    8.4 Mbit   still impossible
//     full table,  2048 positions    4.2 Mbit   fits, but spends 80%
//                                               of all BRAM to cap
//                                               context at 2048
//     incremental (chosen)          ~70 kbit    full 40960 context
//
//   1205x smaller, and the only option that reaches the model's real
//   context length at all. Not a close call, which is why it is worth
//   recording the numbers rather than the conclusion.
//
//   PHASE IS A 32-BIT FRACTION OF ONE TURN, so modulo-2*pi is just
//   integer overflow - no compare, no subtract. Accumulation is
//   integer addition and therefore EXACT: after n positions the phase
//   is exactly n*delta_i mod 2^32, with no drift. The only error is
//   the one-time rounding of delta_i, which accumulates linearly to
//   4.8e-6 turns = 3.0e-5 rad after 40960 positions. The tightest
//   delta is delta_63 = 848 (5.9e-4 relative), and that is what makes
//   the phase 32 bits rather than 24.
//
//   SEEK. Pure accumulation only reaches position n by stepping n
//   times, which is fine for prefill and decode - both visit
//   positions in order - but not for restarting mid-sequence or
//   reusing a cached KV prefix. So there is also a seek path:
//   phase_i = pos * delta_i, one multiply per frequency, 64 cycles,
//   O(1) in the position. Accumulate is the common case; seek exists
//   so an out-of-order start does not cost 40960 steps.
//
//   ONE SIN TABLE SERVES BOTH. cos(p) = sin(p + quarter turn), so cos
//   is the same table read at index + SIN_N/4. Free 2x saving.
//
//   INDEX WRAP IS CORRECT HERE. Reading index+1 wrapping from 4095
//   back to 0 is right, because sin is periodic. That is the OPPOSITE
//   of swiglu_gate, where the identical wrap was a bug and the table
//   needed a 257th entry, because SiLU is not periodic. Same code
//   shape, opposite correct answer.
//
//   The table is replicated across four read ports rather than
//   time-multiplexed: 4 x 65.5 kbit = 262 kbit, about 5% of device
//   BRAM, to get one frequency per cycle instead of four.
//
//====================================================================
// A VERIFICATION TRAP WORTH STATING UP FRONT
//   A phase error does NOT break norm preservation. Rotating by a
//   slightly wrong angle is still a rotation, so |v| is preserved
//   exactly. The norm check that the software pipeline uses can only
//   catch AMPLITUDE error - cos^2+sin^2 != 1 - which means it cannot
//   detect a wrong frequency table, a mis-scaled delta, or a broken
//   accumulator. Those need comparison against the software golden's
//   actual rotated values. Norm preservation is necessary, not
//   sufficient; both checks are needed.
//====================================================================
module rope #(
    parameter int NF      = 64,     // frequency pairs = head_dim/2
    parameter int PHASE_W = 32,
    parameter int SIN_N   = 4096,
    parameter int SIN_W   = 16      // Q1.15
)(
    input  logic clk,
    input  logic rst_n,

    // step every phase forward one position and emit its cos/sin
    input  logic                   start,
    // load an absolute position instead (phase_i = seek_pos * delta_i)
    input  logic                   seek,
    input  logic [PHASE_W-1:0]     seek_pos,
    output logic                   busy,

    // NF (cos, sin) pairs, one per cycle
    output logic                   cs_valid,
    output logic [$clog2(NF)-1:0]  cs_idx,
    output logic signed [SIN_W-1:0] cos_out,
    output logic signed [SIN_W-1:0] sin_out
);

    localparam int FW    = $clog2(NF);
    localparam int SIDX  = $clog2(SIN_N);          // 12
    localparam int QUART = SIN_N / 4;
    localparam logic [FW-1:0] LAST_F = NF - 1;

    if ((1 << SIDX) != SIN_N) begin : gen_sin_pow2
        initial begin
            $error("rope: SIN_N (%0d) must be a power of two", SIN_N);
            $fatal(1);
        end
    end

    //----------------------------------------------------------------
    // Tables. freq is a ROM; phase is read-modify-write state.
    //----------------------------------------------------------------
    (* rom_style = "block" *) logic [PHASE_W-1:0] freq_rom [NF];
    initial $readmemh("rtl/rope_freq.memh", freq_rom);

    logic [PHASE_W-1:0] phase_mem [NF];

    // Four copies of one sin table: sin needs tab[i] and tab[i+1],
    // cos needs the same two at a quarter-turn offset. Replicating
    // buys one frequency per cycle for ~5% of device BRAM.
    (* rom_style = "block" *) logic signed [SIN_W-1:0] sin_a [SIN_N];
    (* rom_style = "block" *) logic signed [SIN_W-1:0] sin_b [SIN_N];
    (* rom_style = "block" *) logic signed [SIN_W-1:0] cos_a [SIN_N];
    (* rom_style = "block" *) logic signed [SIN_W-1:0] cos_b [SIN_N];
    initial begin
        $readmemh("rtl/rope_sin.memh", sin_a);
        $readmemh("rtl/rope_sin.memh", sin_b);
        $readmemh("rtl/rope_sin.memh", cos_a);
        $readmemh("rtl/rope_sin.memh", cos_b);
    end

    //----------------------------------------------------------------
    // FSM: one frequency per cycle, NF cycles per position.
    //----------------------------------------------------------------
    typedef enum logic [1:0] { S_IDLE, S_STEP, S_TAIL } state_t;
    state_t state;

    logic [FW-1:0]      fi;
    logic               is_seek;
    logic [PHASE_W-1:0] pos_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            fi      <= '0;
            is_seek <= 1'b0;
            pos_q   <= '0;
        end else begin
            case (state)
                S_IDLE: if (start || seek) begin
                    state   <= S_STEP;
                    fi      <= '0;
                    is_seek <= seek;
                    pos_q   <= seek_pos;
                end
                S_STEP: if (fi == LAST_F) state <= S_TAIL;
                        else              fi    <= fi + 1'b1;
                S_TAIL: state <= S_IDLE;      // let the last lookup land
                default: state <= S_IDLE;
            endcase
        end
    end

    assign busy = (state != S_IDLE);

    //----------------------------------------------------------------
    // Phase update. Accumulate is exact integer addition; seek is one
    // multiply. Natural PHASE_W overflow is the modulo-2*pi.
    //----------------------------------------------------------------
    logic [PHASE_W-1:0]   delta;
    logic [PHASE_W-1:0]   cur;
    logic [2*PHASE_W-1:0] seek_prod;
    logic [PHASE_W-1:0]   next_phase;

    assign delta      = freq_rom[fi];
    assign cur        = phase_mem[fi];
    assign seek_prod  = pos_q * delta;
    assign next_phase = is_seek ? seek_prod[PHASE_W-1:0] : (cur + delta);

    always_ff @(posedge clk) begin
        if (state == S_STEP) phase_mem[fi] <= next_phase;
    end

    //----------------------------------------------------------------
    // cos/sin lookup with linear interpolation, on the UPDATED phase.
    //----------------------------------------------------------------
    logic [SIDX-1:0] s_idx, c_idx;
    logic [7:0]      frac;

    assign s_idx = next_phase[PHASE_W-1 -: SIDX];
    assign c_idx = s_idx + SIDX'(QUART);
    assign frac  = next_phase[PHASE_W-SIDX-1 -: 8];

    logic signed [SIN_W-1:0] sa, sb, ca, cb;
    logic [7:0]              frac_q;
    logic [FW-1:0]           idx_q;
    logic                    v_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) v_q <= 1'b0;
        else        v_q <= (state == S_STEP);
    end

    always_ff @(posedge clk) begin
        sa     <= sin_a[s_idx];
        sb     <= sin_b[s_idx + SIDX'(1)];   // wraps, and that is correct
        ca     <= cos_a[c_idx];
        cb     <= cos_b[c_idx + SIDX'(1)];
        frac_q <= frac;
        idx_q  <= fi;
    end

    logic signed [SIN_W:0]   sdiff, cdiff;
    logic signed [SIN_W+8:0] sint, cint;

    assign sdiff = {sb[SIN_W-1], sb} - {sa[SIN_W-1], sa};
    assign cdiff = {cb[SIN_W-1], cb} - {ca[SIN_W-1], ca};
    assign sint  = sdiff * $signed({1'b0, frac_q});
    assign cint  = cdiff * $signed({1'b0, frac_q});

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) cs_valid <= 1'b0;
        else        cs_valid <= v_q;
    end

    always_ff @(posedge clk) begin
        sin_out <= sa + SIN_W'(sint >>> 8);
        cos_out <= ca + SIN_W'(cint >>> 8);
        cs_idx  <= idx_q;
    end

endmodule
