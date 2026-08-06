`timescale 1ns / 1ps
//====================================================================
// kv_pack_int8: the K cache line writer, after the end-to-end study
// moved K from 2-bit+QJL to plain INT8.
//
//====================================================================
// WHY K IS NO LONGER ROTATED OR CODEBOOK-QUANTIZED
//   Measured on the real model, 1024-token corpus, 240 evaluation
//   points, against an exact-cache reference:
//
//     K 2b+QJL / V 3b    top-1 53.3%   KL 0.8817   ppl 39.97
//     K INT8   / V 3b    top-1 91.2%   KL 0.0407   ppl 17.56
//     K INT8   / V INT8  top-1 92.1%   KL 0.0373   ppl 17.81
//
//   K at INT8 recovers 37.9 points of top-1 for 1.67x the bytes, and
//   the last 0.9 points would cost a further 1.42x - a real knee. V
//   stays at 3 bits because it costs only 0.9 points there.
//
//   The configuration that was measured quantizes K in the ORIGINAL
//   domain: the study's INT8 path applies H twice, and H*H = N*I, so
//   the rotation cancels exactly. K therefore needs no rotation on the
//   write side and no query rotation on the read side. The rotation
//   existed to flatten outliers for LOW-BIT quantization; at 8 bits
//   there is enough range that it earns nothing.
//
//   This is why K bypasses kv_rotate and kv_quantize entirely rather
//   than reparameterizing them. Feeding an 8-bit uniform codebook to
//   kv_quantize would instead give per-vector normalized 8-bit in the
//   ROTATED domain - plausibly better, but NOT the configuration that
//   was measured, and this project does not ship unmeasured variants.
//
//====================================================================
// FORMAT - deliberately decodable by the EXISTING kv_unpack
//   Exactly 32 words of packed INT8 - 4 coordinates per word,
//   coordinate 0 in the most significant byte of word 0, matching
//   kv_quantize's convention. 128 B/vector against the old 14 words.
//
//   NO per-vector norm word. INT8 uses a per-tensor calibration scale,
//   so there is nothing to put there, and omitting it keeps the line
//   at 32 words - which matters, because w_idx is 5 bits throughout
//   the write path. A 33rd word wraps the index to 0 and silently
//   overwrites word 0. kv_unpack gains a HAS_NORM parameter instead.
//====================================================================
module kv_pack_int8 #(
    parameter int N = 128
)(
    input  logic clk,
    input  logic rst_n,

    // INT8 coordinates in, one per cycle, same handshake kv_quantize used
    input  logic               in_valid,
    input  logic signed [23:0] in_data,     // only [7:0] is significant
    output logic               busy,

    // packed cache line out
    output logic               w_valid,
    output logic [4:0]         w_idx,
    output logic [31:0]        w_data,
    output logic               done
);

    localparam int AW     = $clog2(N);
    localparam int NWQ    = (N * 8) / 32;      // 32
    localparam int NWORDS = NWQ;               // exactly 32 words
    localparam int QB     = N * 8;

    localparam logic [AW:0] N_LAST = N - 1;

    typedef enum logic [1:0] { P_IDLE, P_ACC, P_OUT } state_t;
    state_t state;

    logic [AW:0]   cnt;
    logic [QB-1:0] bits;

    assign busy    = (state != P_IDLE);
    assign w_valid = (state == P_OUT);
    assign w_idx   = cnt[4:0];

    logic [4:0] wsel;
    assign wsel = NWQ[4:0] - 5'd1 - cnt[4:0];

    assign w_data = bits[wsel * 32 +: 32];

    always_ff @(posedge clk) begin
        if (in_valid && (state == P_IDLE || state == P_ACC))
            bits <= {bits[QB-9:0], in_data[7:0]};
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= P_IDLE;
            cnt   <= '0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                P_IDLE: if (in_valid) begin
                    state <= P_ACC;
                    cnt   <= 1;
                end
                P_ACC: if (in_valid) begin
                    if (cnt == N_LAST) begin
                        state <= P_OUT;
                        cnt   <= '0;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end
                P_OUT: if (cnt == NWORDS - 1) begin
                    state <= P_IDLE;
                    cnt   <= '0;
                    done  <= 1'b1;
                end else begin
                    cnt <= cnt + 1'b1;
                end
                default: state <= P_IDLE;
            endcase
        end
    end

endmodule
