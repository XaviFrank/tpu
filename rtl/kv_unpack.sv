`timescale 1ns / 1ps
//====================================================================
// kv_unpack: one packed KV cache line -> raw quantizer indices and
// the per-entry scalars. Extracted from kv_dequant so the streaming
// attention scan and the verification dequantizer share one decoder.
//
//====================================================================
// WHY THIS IS THE BOUNDARY - indices, not reconstructed values
//   kv_dequant reconstructs x-hat in the ORIGINAL domain, which costs
//   a 448-cycle inverse butterfly per entry. In a scan over the whole
//   cache that is 411 M cycles/token at ctx 4096 against a 14.2 M
//   budget - 29x the entire token. The scan therefore cannot consume
//   kv_dequant, and the reason is structural rather than an interface
//   detail:
//
//     K   the rotation CANCELS. Scores are inner products and the
//         rotation is orthogonal, so (H*D*Q).(H*D*K) = N*(Q.K)
//         exactly. Rotate the QUERY once; K is never inverse-rotated.
//     V   the rotation is LINEAR, so it commutes with the weighted
//         sum: sum_j w_j*y_Vj = H*D*(sum_j w_j*V_j). Inverse-rotate
//         once at the end of a scan, not once per position.
//
//   And the scan does not want reconstructed VALUES either, because
//   the level lookup folds into products precomputed once per layer
//   (see docs/attention_read_design_notes.md section 3). So the useful
//   output of a decoder is the raw 2- or 3-bit codes plus the
//   per-entry scalars, and level rescaling belongs on kv_dequant's
//   verification path only.
//
//   codes and signs are exposed as flat buses rather than a streaming
//   port on purpose: kv_dequant slices one coordinate per cycle, the
//   scan slices 16, and a flat bus serves both without either side
//   negotiating a width.
//
//   Word order is exactly what kv_quantize emits, and coordinate 0
//   lands in the MOST significant bits of word 0.
//====================================================================
module kv_unpack #(
    parameter int N      = 128,
    parameter int QBITS  = 3,
    parameter int QJL_EN = 0,
    // INT8 K lines carry no per-vector norm - they use a
    // per-tensor calibration scale - and dropping the slot keeps
    // the line at exactly 32 words, so w_idx stays 5 bits.
    parameter int HAS_NORM = 1
)(
    input  logic clk,
    input  logic rst_n,

    input  logic        w_valid,
    input  logic [31:0] w_data,
    output logic        busy,

    output logic [N*QBITS-1:0] codes,
    output logic [N-1:0]       signs,
    output logic [29:0]        vnorm,
    output logic [29:0]        rnorm,
    output logic               entry_valid
);

    localparam int NWQ    = (N * QBITS) / 32;
    localparam int NWORDS = NWQ + (HAS_NORM != 0 ? 1 : 0)
                                + (QJL_EN  != 0 ? 5 : 0);
    localparam int QB     = N * QBITS;

    typedef enum logic [0:0] { U_IDLE, U_LOAD } state_t;
    state_t state;

    logic [5:0] cnt;

    assign busy = (state != U_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= U_IDLE;
            cnt         <= 6'd0;
            entry_valid <= 1'b0;
        end else begin
            entry_valid <= 1'b0;
            case (state)
                U_IDLE: if (w_valid) begin
                    state <= U_LOAD;
                    cnt   <= 6'd1;
                end
                U_LOAD: if (w_valid) begin
                    if (cnt == NWORDS - 1) begin
                        state       <= U_IDLE;
                        cnt         <= 6'd0;
                        // the last word registers on this same edge,
                        // so the buses are complete when this pulses
                        entry_valid <= 1'b1;
                    end else begin
                        cnt <= cnt + 6'd1;
                    end
                end
                default: state <= U_IDLE;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (w_valid) begin
            if (cnt < NWQ)           codes <= {codes[QB-33:0], w_data};
            else if (cnt == NWQ)     vnorm <= w_data[29:0];
            else if (cnt <= NWQ + 4) signs <= {signs[N-33:0], w_data};
            else                     rnorm <= w_data[29:0];
        end
    end

endmodule
