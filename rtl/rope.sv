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
    parameter int SIN_W   = 16,     // Q1.15
    // See rmsnorm.sv for why these are parameters. Two files, but
    // ROPE_SIN_PATH feeds four arrays (sin_a/sin_b/cos_a/cos_b).
    // Vivado overrides:
    //   -generic ROPE_FREQ_PATH=C:/dev/tpu/rtl/rope_freq.memh
    //   -generic ROPE_SIN_PATH=C:/dev/tpu/rtl/rope_sin.memh
    parameter ROPE_FREQ_PATH = "rtl/rope_freq.memh",
    parameter ROPE_SIN_PATH  = "rtl/rope_sin.memh"
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
    output logic signed [SIN_W-1:0] sin_out,

    // ---- rotation ----
    // Applies the CURRENT position's cos/sin to one head's vector.
    // x is split as the model splits it: x1 = x[0:NF], x2 = x[NF:2*NF],
    // the rotate-half convention already confirmed against real HF
    // source in the software work.
    input  logic                   rot_start,
    output logic                   x_rd_en,
    output logic [$clog2(NF)-1:0]  x_rd_addr,
    input  logic signed [7:0]      x1_rd_data,   // x[j]
    input  logic signed [7:0]      x2_rd_data,   // x[j+NF]
    output logic                   rot_valid,
    output logic [$clog2(NF)-1:0]  rot_idx,
    output logic signed [7:0]      rot_x1,
    output logic signed [7:0]      rot_x2,

    // ---- completion ----
    // busy IS NOT A SAFE COMPLETION SIGNAL and these are not sugar.
    // Two separate hazards:
    //
    //   1. busy does not RISE until the cycle after start/seek/
    //      rot_start is sampled, so "assert rot_start; wait !busy"
    //      reads the gap as already-done and falls straight through.
    //      That is the trap attn_kv_top's T_WWAIT had to be written
    //      around by hand.
    //   2. worse, on the ROTATE path busy falls one cycle EARLY.
    //      rot_valid is (state == S_ROT) delayed twice, so the last
    //      of the NF beats lands one cycle after the FSM has already
    //      returned to S_IDLE. A caller that waits for !busy and then
    //      stops reading loses the final (x1, x2) pair - silently,
    //      and only on the last frequency index.
    //
    // Both pulse in the cycle the FSM is back in S_IDLE, i.e. after
    // every beat has landed and the module will accept a new command.
    // rot_done comes from S_RTAIL, not S_TAIL: S_TAIL is the PHASE
    // step's drain state on the other branch of the FSM, and a
    // rot_done wired to it would pulse on start/seek and never on
    // rot_start.
    output logic                   step_done,
    output logic                   rot_done
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
    initial $readmemh(ROPE_FREQ_PATH, freq_rom);

    logic [PHASE_W-1:0] phase_mem [NF];

    // Four copies of one sin table: sin needs tab[i] and tab[i+1],
    // cos needs the same two at a quarter-turn offset. Replicating
    // buys one frequency per cycle for ~5% of device BRAM.
    (* rom_style = "block" *) logic signed [SIN_W-1:0] sin_a [SIN_N];
    (* rom_style = "block" *) logic signed [SIN_W-1:0] sin_b [SIN_N];
    (* rom_style = "block" *) logic signed [SIN_W-1:0] cos_a [SIN_N];
    (* rom_style = "block" *) logic signed [SIN_W-1:0] cos_b [SIN_N];
    initial begin
        $readmemh(ROPE_SIN_PATH, sin_a);
        $readmemh(ROPE_SIN_PATH, sin_b);
        $readmemh(ROPE_SIN_PATH, cos_a);
        $readmemh(ROPE_SIN_PATH, cos_b);
    end

    //----------------------------------------------------------------
    // FSM: one frequency per cycle, NF cycles per position.
    //----------------------------------------------------------------
    typedef enum logic [2:0] { S_IDLE, S_STEP, S_TAIL, S_ROT, S_RTAIL } state_t;
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
                S_IDLE: if (rot_start) begin
                    state <= S_ROT;
                    fi    <= '0;
                end else if (start || seek) begin
                    state   <= S_STEP;
                    fi      <= '0;
                    is_seek <= seek;
                    pos_q   <= seek_pos;
                end
                S_STEP: if (fi == LAST_F) state <= S_TAIL;
                        else              fi    <= fi + 1'b1;
                S_TAIL: state <= S_IDLE;      // let the last lookup land
                S_ROT:  if (fi == LAST_F) state <= S_RTAIL;
                        else              fi    <= fi + 1'b1;
                S_RTAIL: state <= S_IDLE;     // drain the rotate pipe
                default: state <= S_IDLE;
            endcase
        end
    end

    assign busy = (state != S_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            step_done <= 1'b0;
            rot_done  <= 1'b0;
        end else begin
            step_done <= (state == S_TAIL);
            rot_done  <= (state == S_RTAIL);
        end
    end

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

    //----------------------------------------------------------------
    // Keep this position's cos/sin so the rotation can replay them
    // across all 24 heads without regenerating the phase each time.
    //----------------------------------------------------------------
    logic signed [SIN_W-1:0] cos_mem [NF];
    logic signed [SIN_W-1:0] sin_mem [NF];

    always_ff @(posedge clk) begin
        if (cs_valid) begin
            cos_mem[cs_idx] <= cos_out;
            sin_mem[cs_idx] <= sin_out;
        end
    end

    //----------------------------------------------------------------
    // Rotation datapath. One (x1, x2) pair per cycle, NF cycles/head.
    //
    //   rot_x1 = x1*cos - x2*sin
    //   rot_x2 = x2*cos + x1*sin
    //
    // NO REQUANT TAP HERE, and that is deliberate rather than an
    // omission: a rotation preserves magnitude, so the output carries
    // the SAME scale as the input. Every other unit in this design
    // needs requant because it changes scale; this one does not. All
    // that is required is the Q1.15 shift with round-half-up, matching
    // the convention requant itself uses, and a saturate.
    //
    // Saturation is real, not theoretical: a single component of a
    // rotated vector can exceed the original component maximum by up
    // to sqrt(2) even though the norm is unchanged, so INT8 in does
    // not guarantee INT8 out. The TB measures how often that actually
    // bites on real data rather than assuming it is rare.
    //----------------------------------------------------------------
    logic signed [SIN_W-1:0] c_r, s_r;
    logic                    rv1, rv2;
    logic [FW-1:0]           ri1, ri2;
    logic signed [24:0]      r1, r2;
    logic signed [25:0]      r1_rnd, r2_rnd;

    assign x_rd_en   = (state == S_ROT);
    assign x_rd_addr = fi;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rv1 <= 1'b0;
            rv2 <= 1'b0;
        end else begin
            rv1 <= (state == S_ROT);
            rv2 <= rv1;
        end
    end

    // cos/sin are REGISTERED here so they land in the same cycle as the
    // external buffer's 1-cycle-latency read data. Reading them
    // combinationally would present cos[j] against x[j-1].
    always_ff @(posedge clk) begin
        c_r <= cos_mem[fi];
        s_r <= sin_mem[fi];
        ri1 <= fi;
        ri2 <= ri1;
        r1  <= x1_rd_data * c_r - x2_rd_data * s_r;
        r2  <= x2_rd_data * c_r + x1_rd_data * s_r;
    end

    assign r1_rnd = 26'(r1) + 26'sd16384;      // round half up at Q1.15
    assign r2_rnd = 26'(r2) + 26'sd16384;

    always_comb begin
        if      ((r1_rnd >>> 15) >  26'sd127)  rot_x1 = 8'sd127;
        else if ((r1_rnd >>> 15) < -26'sd128)  rot_x1 = -8'sd128;
        else                                   rot_x1 = 8'((r1_rnd >>> 15));
        if      ((r2_rnd >>> 15) >  26'sd127)  rot_x2 = 8'sd127;
        else if ((r2_rnd >>> 15) < -26'sd128)  rot_x2 = -8'sd128;
        else                                   rot_x2 = 8'((r2_rnd >>> 15));
    end

    assign rot_valid = rv2;
    assign rot_idx   = ri2;

endmodule
