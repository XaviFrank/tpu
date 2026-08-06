`timescale 1ns / 1ps
//====================================================================
// kv_quantize: normalize, low-bit quantize, and pack one rotated
// K or V vector. Step 3 of the KV cache write path.
//
//====================================================================
// NO DIVISION, AND NO ROUNDING CONVENTION, IN THE COMPARE PATH
//   The quantizer wants q_i = codebook(y_i / ||y||), which reads like
//   it needs a reciprocal per coordinate. It does not. Comparing
//   y_i/||y|| against a constant t_k is the same as comparing y_i
//   against t_k*||y||, so the norm is folded into the thresholds ONCE
//   per vector - NLEV-1 multiplies - and every one of the N
//   coordinates is then a plain integer compare. The reconstruction
//   levels get the same treatment, which is what makes the residual
//   exact too.
//
//   ||y|| is a genuine integer square root (restoring, one bit-pair
//   per cycle, ROOT_W cycles), not a Newton-Raphson approximation like
//   rmsnorm's rsqrt. That difference is deliberate: rmsnorm needed a
//   reciprocal square root in a throughput path, whereas this is ONE
//   root per vector, so exactness is worth more than the cycles. It
//   keeps the whole datapath integer-exact, which is what lets the
//   testbench demand bit-exact agreement instead of a tolerance.
//
//====================================================================
// ONE CODEBOOK PER ROTATION MODE - required, not tidiness
//   A Haar rotation makes coordinates ~Gaussian, which is the Beta
//   density TurboQuant's Lloyd-Max quantizer is fitted to. The
//   Hadamard path spreads energy deterministically and comes out with
//   a different, flatter distribution. Scoring one with the other's
//   codebook charges it for a mismatch that has nothing to do with the
//   rotation. CB_PATH therefore selects a codebook fitted to the mode
//   actually in use; tb/gen_kv_codebooks.py fits all four.
//
//====================================================================
// ASYMMETRIC FORMAT - K and V are consumed differently
//   QBITS=3, QJL_EN=0   V: pure MSE. V is consumed as a
//                       probability-weighted sum, so reconstruction
//                       error is exactly the right metric.
//   QBITS=2, QJL_EN=1   K: MSE stage plus a 1-bit quantized-JL
//                       residual and its norm. MSE-optimal quantizers
//                       are BIASED for inner products - the paper's
//                       own stated reason for the QJL stage - and
//                       attention scores ARE inner products.
//
//   This is why "write side only" does not defer the question: the
//   write path fixes the format the read side has to live with. A
//   pure-MSE format would also make a reconstruction-error testbench
//   look excellent while being the wrong format for attention.
//
//====================================================================
// PACKING - groups that divide evenly, so no bits are wasted
//   N*QBITS is a multiple of 32 for both configurations here
//   (128*3 = 384 = 12 words, 128*2 = 256 = 8 words), so indices pack
//   bit-continuously with ZERO padding. Element 0 occupies the most
//   significant bits of word 0. Word map:
//
//     0 .. NWQ-1      packed quantizer indices
//     NWQ             ||y||, the vector norm
//     NWQ+1 .. NWQ+4  QJL residual sign bits      (QJL_EN only)
//     NWQ+5           ||residual||                (QJL_EN only)
//
//   V costs 13 words per 128 coords = 3.25 bits/coord, K costs 14 =
//   3.5 bits/coord. The norms are the entire overhead over the nominal
//   bit budget, and 3.5 bits/channel is exactly where the paper
//   reports quality neutrality.
//====================================================================
module kv_quantize #(
    parameter int N      = 128,
    parameter int QBITS  = 3,
    parameter int QJL_EN = 0,
    // Vivado: -generic CB_PATH=C:/dev/tpu/rtl/kv_cb_m0_b3.memh
    parameter CB_PATH = "rtl/kv_cb_m0_b3.memh"
)(
    input  logic clk,
    input  logic rst_n,

    // rotated vector in, one coordinate per cycle
    input  logic               in_valid,
    input  logic signed [23:0] in_data,
    output logic               busy,

    // packed cache line out, one 32-bit word per cycle
    output logic               w_valid,
    output logic [4:0]         w_idx,
    output logic [31:0]        w_data,
    output logic               done
);

    localparam int AW     = $clog2(N);
    localparam int NLEV   = 1 << QBITS;
    localparam int NWQ    = (N * QBITS) / 32;
    localparam int NWORDS = (QJL_EN != 0) ? (NWQ + 6) : (NWQ + 1);
    localparam int SOS_W  = 60;
    localparam int ROOT_W = 30;
    localparam int QB     = N * QBITS;

    localparam logic [AW:0] N_LAST = N - 1;

    if ((N * QBITS) % 32 != 0) begin : gen_pack_align
        initial begin
            $error("kv_quantize: N*QBITS (%0d) must be a multiple of 32",
                   N * QBITS);
            $fatal(1);
        end
    end

    (* rom_style = "distributed" *) logic [15:0] cb_rom [2*NLEV-1];
    initial $readmemh(CB_PATH, cb_rom);

    // Array-element signedness is unreliable here, so every signed
    // compare goes through a function argument - a plain variable.
    function automatic logic sge(input logic [31:0] a, input logic [31:0] b);
        sge = ($signed(a) >= $signed(b));
    endfunction

    typedef enum logic [2:0] {
        S_IDLE, S_ACC, S_SQRT, S_THR, S_QUANT, S_RSQRT, S_OUT
    } state_t;
    state_t state;

    logic [AW:0]       cnt;
    logic [SOS_W-1:0]  sos, rsos;
    logic [23:0]       ymem [N];
    logic [31:0]       thr_s [NLEV-1];
    logic [31:0]       lev_s [NLEV];
    logic [ROOT_W-1:0] vnorm, rnorm;
    logic [QB-1:0]     qbits;
    logic [N-1:0]      sbits;

    assign busy = (state != S_IDLE);

    //----------------------------------------------------------------
    // Shared restoring integer square root, run twice: once for the
    // vector norm, once for the residual norm. Exact floor(sqrt(n)).
    //----------------------------------------------------------------
    logic [SOS_W-1:0]  sq_n;
    logic [ROOT_W-1:0] sq_q;
    logic [ROOT_W+3:0] sq_r, sq_r2, sq_t;
    logic [4:0]        sq_i;
    logic [5:0]        sq_sh;
    logic [1:0]        sq_bits;
    logic              sq_go, sq_busy, sq_rdy;

    assign sq_sh   = {1'b0, sq_i} << 1;
    assign sq_bits = (sq_n >> sq_sh) & 2'b11;
    assign sq_r2   = (sq_r << 2) | {{(ROOT_W+2){1'b0}}, sq_bits};
    assign sq_t    = {2'b00, sq_q, 2'b01};
    assign sq_rdy  = !sq_busy && !sq_go;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sq_busy <= 1'b0;
            sq_i    <= 5'd0;
        end else if (sq_go) begin
            sq_busy <= 1'b1;
            sq_i    <= ROOT_W - 1;
        end else if (sq_busy) begin
            if (sq_i == 5'd0) sq_busy <= 1'b0;
            else              sq_i    <= sq_i - 5'd1;
        end
    end

    always_ff @(posedge clk) begin
        if (sq_go) begin
            sq_q <= '0;
            sq_r <= '0;
        end else if (sq_busy) begin
            if (sq_r2 >= sq_t) begin
                sq_r <= sq_r2 - sq_t;
                sq_q <= (sq_q << 1) | 1'b1;
            end else begin
                sq_r <= sq_r2;
                sq_q <= sq_q << 1;
            end
        end
    end

    //----------------------------------------------------------------
    // Fold ||y|| into the codebook: one multiply per cycle in S_THR.
    //----------------------------------------------------------------
    logic signed [15:0] cb_cur;
    logic signed [47:0] cb_prod;
    logic [31:0]        cb_scaled;

    assign cb_cur    = cb_rom[cnt[4:0]];
    assign cb_prod   = cb_cur * $signed({1'b0, vnorm});
    assign cb_scaled = cb_prod[46:15];

    //----------------------------------------------------------------
    // Quantize: compare against the scaled thresholds, compare only.
    //----------------------------------------------------------------
    logic [23:0]        y_cur;
    logic [31:0]        y_ext, rec;
    logic signed [31:0] resid;
    logic [63:0]        rsq;
    logic [47:0]        in_sq;
    logic [QBITS-1:0]   qsel;
    integer             kk;

    assign y_cur = ymem[cnt[AW-1:0]];
    assign y_ext = {{8{y_cur[23]}}, y_cur};
    assign in_sq = in_data * in_data;

    // Thresholds ascend, so counting the ones at or below the
    // coordinate IS the codeword index. Counting rather than assigning
    // k+1 also keeps a part-select of a loop variable out of an
    // always_comb, which Icarus only half-supports.
    always_comb begin
        qsel = '0;
        for (kk = 0; kk < NLEV-1; kk = kk + 1)
            if (sge(y_ext, thr_s[kk])) qsel = qsel + 1'b1;
    end

    assign rec   = lev_s[qsel];
    assign resid = $signed(y_ext) - $signed(rec);
    assign rsq   = resid * resid;

    //----------------------------------------------------------------
    // Output word mux. Element 0 lands in the top bits of word 0.
    //----------------------------------------------------------------
    // Word selects hoisted to continuous assigns for the same Icarus
    // reason as the codebook indices above.
    logic [4:0] qwsel, swsel;
    assign qwsel = NWQ[4:0] - 5'd1 - cnt[4:0];
    assign swsel = NWQ[4:0] + 5'd4 - cnt[4:0];

    assign w_valid = (state == S_OUT);
    assign w_idx   = cnt[4:0];

    always_comb begin
        if (cnt < NWQ)
            w_data = qbits[qwsel * 32 +: 32];
        else if (cnt == NWQ)
            w_data = {{(32-ROOT_W){1'b0}}, vnorm};
        else if (cnt <= NWQ + 4)
            w_data = sbits[swsel * 32 +: 32];
        else
            w_data = {{(32-ROOT_W){1'b0}}, rnorm};
    end

    //----------------------------------------------------------------
    // Sequencer
    //----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            cnt   <= '0;
            sq_go <= 1'b0;
            done  <= 1'b0;
        end else begin
            sq_go <= 1'b0;
            done  <= 1'b0;
            case (state)
                S_IDLE: if (in_valid) begin
                    state <= S_ACC;
                    cnt   <= 1;
                end
                S_ACC: if (in_valid) begin
                    if (cnt == N_LAST) begin
                        state <= S_SQRT;
                        sq_go <= 1'b1;
                        cnt   <= '0;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end
                S_SQRT: if (sq_rdy) begin
                    state <= S_THR;
                    cnt   <= '0;
                end
                S_THR: if (cnt == 2*NLEV - 2) begin
                    state <= S_QUANT;
                    cnt   <= '0;
                end else begin
                    cnt <= cnt + 1'b1;
                end
                S_QUANT: if (cnt == N_LAST) begin
                    cnt <= '0;
                    if (QJL_EN != 0) begin
                        state <= S_RSQRT;
                        sq_go <= 1'b1;
                    end else begin
                        state <= S_OUT;
                    end
                end else begin
                    cnt <= cnt + 1'b1;
                end
                S_RSQRT: if (sq_rdy) begin
                    state <= S_OUT;
                    cnt   <= '0;
                end
                S_OUT: if (cnt == NWORDS - 1) begin
                    state <= S_IDLE;
                    cnt   <= '0;
                    done  <= 1'b1;
                end else begin
                    cnt <= cnt + 1'b1;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // Sum of squares and vector capture, as the rotation streams in.
    always_ff @(posedge clk) begin
        if (state == S_IDLE) begin
            if (in_valid) sos <= {{(SOS_W-48){1'b0}}, in_sq};
            else          sos <= '0;
        end else if (state == S_ACC && in_valid) begin
            sos <= sos + {{(SOS_W-48){1'b0}}, in_sq};
        end
    end

    always_ff @(posedge clk) begin
        if (in_valid && (state == S_IDLE || state == S_ACC))
            ymem[cnt[AW-1:0]] <= in_data;
    end

    // The square-root operand, loaded on the cycle sq_go is raised.
    always_ff @(posedge clk) begin
        if (state == S_ACC && in_valid && cnt == N_LAST)
            sq_n <= sos + {{(SOS_W-48){1'b0}}, in_sq};
        else if (state == S_QUANT && cnt == N_LAST)
            sq_n <= rsos + rsq[SOS_W-1:0];
    end

    always_ff @(posedge clk) begin
        if (state == S_SQRT  && sq_rdy) vnorm <= sq_q;
        if (state == S_RSQRT && sq_rdy) rnorm <= sq_q;
    end

    // Index hoisted out of the always_ff: a part-select used as an
    // array index inside a clocked process makes Icarus sensitive to
    // the whole vector and emit a "sorry" note.
    logic [4:0] cbi, lvi;
    assign cbi = cnt[4:0];
    assign lvi = cnt[4:0] - NLEV[4:0] + 5'd1;

    always_ff @(posedge clk) begin
        if (state == S_THR) begin
            if (cnt < NLEV - 1) thr_s[cbi] <= cb_scaled;
            else                lev_s[lvi] <= cb_scaled;
        end
    end

    always_ff @(posedge clk) begin
        if (state == S_QUANT) begin
            qbits <= {qbits[QB-QBITS-1:0], qsel};
            sbits <= {sbits[N-2:0], resid[31]};
        end
    end

    always_ff @(posedge clk) begin
        if (state == S_THR)        rsos <= '0;
        else if (state == S_QUANT) rsos <= rsos + rsq[SOS_W-1:0];
    end

endmodule
