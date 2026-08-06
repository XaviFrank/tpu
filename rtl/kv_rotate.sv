`timescale 1ns / 1ps
//====================================================================
// kv_rotate: the rotation stage of the KV cache quantized write path.
//
//   Flattens outlier magnitude before low-bit quantization, so a fixed
//   scalar codebook does not have to cope with a few large values.
//   This is step 2 of the spec; normalize/quantize/pack is step 3 and
//   deliberately not here.
//
//====================================================================
// WHICH ROTATION - the answer is NOT the cheap one, and that matters
//====================================================================
//   TurboQuant (arXiv 2504.19874) specifies a DENSE random orthogonal
//   matrix, "generate Pi by applying QR decomposition on a random
//   matrix with i.i.d Normal entries", and its guarantee rests on Pi*x
//   being uniform on the unit sphere. The words "Hadamard", "Walsh"
//   and "O(d log d)" appear NOWHERE in the paper. The widely-repeated
//   claim that it uses a fast Walsh-Hadamard transform comes from
//   TurboQuant-H, a third-party variant whose own justification -
//   "for offline weight quantization we don't need the data-oblivious
//   guarantees of a random rotation" - excludes KV cache, which is
//   precisely the online data-oblivious case.
//
//   So ROT_MODE is not a convenience switch. Mode 1 is what the paper
//   says; mode 0 is the cheap transform the literature is widely
//   believed to say. Both are built because the question that actually
//   governs the datapath is not "what does the paper specify" - that
//   is settled - but "does the cheap one lose accuracy ON THIS DATA,
//   IN INT8, AT d=128", which is measurable here and is measured in
//   the round-trip TB. Full reasoning in docs/kv_cache_design_notes.md.
//
//     ROT_MODE = 0   randomized Hadamard, y = H_unnorm * (D * x)
//                    7 butterfly stages, adds/subtracts only.
//                    NO MULTIPLIER, and 128 bits of table.
//     ROT_MODE = 1   dense Pi, y = Pi_int8 * x
//                    N*N MACs, 16 KB of table, one DSP.
//
//   The two live in separate generate branches so a mode-0 build
//   really does carry no multiplier and no Pi ROM, rather than relying
//   on synthesis to prove the unused half away.
//
//====================================================================
// NO SCALE, AND NO ROUNDING, IN EITHER PATH
//   H_unnorm has +-1 entries, so mode 0 is exact integer arithmetic
//   end to end - no rounding convention to get wrong, and
//   H_unnorm^T * H_unnorm = N*I EXACTLY, so it is losslessly
//   invertible. Mode 1 is INT8 x INT8 summed in 24 bits, also exact.
//   The 1/sqrt(N) that would make H orthonormal, and Pi's own
//   quantization scale, are BOTH deferred to the normalize step, where
//   one per-vector norm absorbs them. A rotation correct only up to
//   scale is still a correct rotation for quantization purposes, and
//   keeping scale out of the datapath is what lets the testbench
//   demand bit-exact agreement instead of a tolerance.
//
//   Width: |x| <= 128 and N = 128 doublings give |y| <= 2^14, so 24
//   bits is generous for mode 0. Mode 1 needs 8+8+log2(128) = 23 plus
//   sign. One width serves both.
//
//====================================================================
// KNOWN AREA CHARACTERISTIC, stated rather than discovered later
//   The mode-0 butterfly holds the vector in a register array read at
//   two arbitrary indices and written at two per cycle. That is a
//   4-port register file: it will infer flip-flops plus wide muxes,
//   not LUTRAM (no dual-write LUTRAM exists) and not BRAM. Expect
//   ~3 kbit of FF and a few thousand LUTs of mux at N=128. If Vivado
//   says that is too expensive, the fix is to serialize the butterfly
//   over banks, which costs cycles this design has in abundance -
//   448 rotations per token against an ~8 ms token budget. Recorded
//   now so a large LUT number reads as expected, not as a surprise.
//====================================================================
module kv_rotate #(
    parameter int N        = 128,   // head_dim
    parameter int ROT_MODE = 0,     // 0 = randomized Hadamard, 1 = dense Pi
    // Paths are parameters, not bare filenames: $readmemh resolves
    // against the CWD and Vivado's synth run has a different one than
    // regress.sh. Same lesson as rmsnorm/swiglu_gate/rope. Vivado:
    //   -generic ROT_SIGN_PATH=C:/dev/tpu/rtl/kv_rot_sign.memh
    //   -generic ROT_PI_PATH=C:/dev/tpu/rtl/kv_rot_pi.memh
    parameter ROT_SIGN_PATH = "rtl/kv_rot_sign.memh",
    parameter ROT_PI_PATH   = "rtl/kv_rot_pi.memh"
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                 start,
    output logic                 busy,

    // input vector, external buffer, 1-cycle read latency
    output logic                 x_rd_en,
    output logic [$clog2(N)-1:0] x_rd_addr,
    input  logic signed [7:0]    x_rd_data,

    // rotated vector, one element per cycle
    output logic                 y_valid,
    output logic [$clog2(N)-1:0] y_idx,
    output logic signed [23:0]   y_out,
    output logic                 done
);

    localparam int AW     = $clog2(N);
    localparam int ACC_W  = 24;
    localparam int STAGES = AW;             // log2(N) butterfly stages
    localparam int HALF   = N / 2;

    localparam logic [AW:0]   N_C     = N[AW:0];
    localparam logic [AW-1:0] ONE     = 1;
    localparam logic [AW-1:0] IJ_LAST = N - 1;

    if ((1 << AW) != N) begin : gen_n_pow2
        initial begin
            $error("kv_rotate: N (%0d) must be a power of two", N);
            $fatal(1);
        end
    end

    //----------------------------------------------------------------
    // Shared skeleton. calc_done and out_data are driven from inside
    // the generate branches, so they are declared at module level.
    //----------------------------------------------------------------
    typedef enum logic [1:0] { S_IDLE, S_LOAD, S_CALC, S_OUT } state_t;
    state_t state;

    logic [AW:0]      lcnt;         // load and output counter, 0..N
    logic             ld_v;
    logic [AW-1:0]    ld_i;
    logic             calc_done;
    logic [ACC_W-1:0] out_data;

    assign x_rd_en   = (state == S_LOAD) && (lcnt < N_C);
    assign x_rd_addr = lcnt[AW-1:0];
    assign busy      = (state != S_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) ld_v <= 1'b0;
        else        ld_v <= x_rd_en;
    end

    always_ff @(posedge clk) ld_i <= lcnt[AW-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            lcnt  <= '0;
        end else begin
            case (state)
                S_IDLE: if (start) begin
                    state <= S_LOAD;
                    lcnt  <= '0;
                end
                // one extra cycle so the last 1-cycle-latency read lands
                S_LOAD: if (lcnt == N_C) begin
                    state <= S_CALC;
                    lcnt  <= '0;
                end else lcnt <= lcnt + 1'b1;
                S_CALC: if (calc_done) begin
                    state <= S_OUT;
                    lcnt  <= '0;
                end
                S_OUT:  if (lcnt == N_C - 1'b1) begin
                    state <= S_IDLE;
                    lcnt  <= '0;
                end else lcnt <= lcnt + 1'b1;
                default: state <= S_IDLE;
            endcase
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y_valid <= 1'b0;
            done    <= 1'b0;
        end else begin
            y_valid <= (state == S_OUT);
            done    <= (state == S_OUT) && (lcnt == N_C - 1'b1);
        end
    end

    always_ff @(posedge clk) begin
        y_out <= out_data;
        y_idx <= lcnt[AW-1:0];
    end

    generate
    if (ROT_MODE == 0) begin : gen_had
        //------------------------------------------------------------
        // Randomized Hadamard: y = H_unnorm * (D * x).
        //
        // D is the only stored table - 128 bits. H itself is the
        // butterfly wiring, not coefficients, which is the whole cost
        // argument against the dense path made concrete.
        //
        // In-place iterative FWHT, one pair per cycle. Stage s pairs
        // index i with i XOR 2^s, over the HALF indices that have bit
        // s clear; the index arithmetic below is "insert a zero bit at
        // position s into the pair counter", which enumerates exactly
        // those. STAGES*HALF = 448 cycles per vector, against a budget
        // of ~8 ms per token for 448 vectors - deliberately the dull
        // serial form, because there is no throughput pressure here.
        //------------------------------------------------------------
        (* rom_style = "distributed" *) logic [7:0] sign_rom [N];
        initial $readmemh(ROT_SIGN_PATH, sign_rom);

        logic [ACC_W-1:0] mem [N];

        logic [ACC_W-1:0] x_ext, x_flip;
        assign x_ext  = {{(ACC_W-8){x_rd_data[7]}}, x_rd_data};
        // negate as ~x+1 rather than unary minus: unpacked-array
        // element signedness is unreliable here, so nothing depends
        // on it anywhere in this file.
        assign x_flip = sign_rom[ld_i][0] ? (~x_ext + 1'b1) : x_ext;

        logic [2:0]      st;
        logic [AW-2:0]   pr;
        logic [AW-1:0]   pr_ext, mask_lo, idx_i, idx_j;

        localparam logic [AW-2:0] PR_LAST = HALF - 1;

        assign pr_ext  = {1'b0, pr};
        assign mask_lo = (ONE << st) - ONE;
        assign idx_i   = ((pr_ext >> st) << (st + 1'b1)) | (pr_ext & mask_lo);
        assign idx_j   = idx_i | (ONE << st);

        logic [ACC_W-1:0] a_val, b_val;
        assign a_val = mem[idx_i];
        assign b_val = mem[idx_j];

        always_ff @(posedge clk) begin
            if (ld_v) begin
                mem[ld_i] <= x_flip;
            end else if (state == S_CALC) begin
                mem[idx_i] <= a_val + b_val;
                mem[idx_j] <= a_val - b_val;
            end
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                st <= 3'd0;
                pr <= '0;
            end else if (state != S_CALC) begin
                st <= 3'd0;
                pr <= '0;
            end else if (pr == PR_LAST) begin
                pr <= '0;
                st <= st + 3'd1;
            end else begin
                pr <= pr + 1'b1;
            end
        end

        assign calc_done = (state == S_CALC) && (st == STAGES - 1)
                                             && (pr == PR_LAST);
        assign out_data  = mem[lcnt[AW-1:0]];

    end else begin : gen_dense
        //------------------------------------------------------------
        // Dense Pi: y[i] = sum_j Pi[i][j] * x[j], one MAC per cycle,
        // N*N = 16384 cycles per vector.
        //
        // THE SERIAL FORM IS FOR MEASUREMENT, NOT DEPLOYMENT. Pi is
        // fixed and data-oblivious, so in the real system this is a
        // fixed-matrix matvec that belongs on ws_array via
        // ws_tiling_engine (128x128 = 64 tiles of 16x16) at zero new
        // DSPs, ~3% of a token. What matters for the ROT_MODE
        // comparison is the NUMERICS, which are identical either way,
        // so this file keeps the small serial version and the design
        // note carries the deployment mapping.
        //
        // The Pi ROM read is REGISTERED. A combinational read of a
        // 16384-entry array cannot infer BRAM and would land 16 KB in
        // LUTs; registering it costs one pipeline stage, drained by
        // ctr_last_r before calc_done fires.
        //------------------------------------------------------------
        (* rom_style = "block" *) logic [7:0] pi_rom [N*N];
        initial $readmemh(ROT_PI_PATH, pi_rom);

        logic [7:0]       xmem [N];
        logic [ACC_W-1:0] ymem [N];

        always_ff @(posedge clk) begin
            if (ld_v) xmem[ld_i] <= x_rd_data;
        end

        logic [AW-1:0]     oi, ij, oi_r, ij_r;
        logic [7:0]        pi_q, x_q;
        logic signed [8:0] pi_s, x_s;
        logic signed [17:0] prod;
        logic [ACC_W-1:0]  acc, prod_ext;
        logic              mac_v, ctr_last, ctr_last_r;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                oi <= '0;
                ij <= '0;
            end else if (state != S_CALC) begin
                oi <= '0;
                ij <= '0;
            end else if (ij == IJ_LAST) begin
                ij <= '0;
                oi <= oi + 1'b1;
            end else begin
                ij <= ij + 1'b1;
            end
        end

        always_ff @(posedge clk) begin
            pi_q <= pi_rom[{oi, ij}];
            x_q  <= xmem[ij];
            oi_r <= oi;
            ij_r <= ij;
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) mac_v <= 1'b0;
            else        mac_v <= (state == S_CALC);
        end

        assign pi_s     = {pi_q[7], pi_q};
        assign x_s      = {x_q[7],  x_q};
        assign prod     = pi_s * x_s;
        assign prod_ext = {{(ACC_W-18){prod[17]}}, prod};

        always_ff @(posedge clk) begin
            if (!mac_v)                acc <= '0;
            else if (ij_r == IJ_LAST)  acc <= '0;
            else                       acc <= acc + prod_ext;
        end

        always_ff @(posedge clk) begin
            if (mac_v && (ij_r == IJ_LAST)) ymem[oi_r] <= acc + prod_ext;
        end

        assign ctr_last = (state == S_CALC) && (oi == IJ_LAST)
                                            && (ij == IJ_LAST);

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) ctr_last_r <= 1'b0;
            else        ctr_last_r <= ctr_last;
        end

        assign calc_done = ctr_last_r;
        assign out_data  = ymem[lcnt[AW-1:0]];
    end
    endgenerate

endmodule
