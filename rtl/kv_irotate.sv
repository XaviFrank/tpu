`timescale 1ns / 1ps
//====================================================================
// kv_irotate: the INVERSE rotation, extracted from kv_dequant so the
// attention output path can use it without dragging the dequantizer
// along.
//
//   Same move as kv_unpack, and for the same reason. kv_dequant is
//   unpack + level rescale + inverse rotation fused into one
//   sequencer; the attention output needs ONLY the third of those,
//   applied to a 48-bit accumulator rather than to a packed cache
//   line. Copying the butterfly would have left two of them to keep in
//   step. This file is the original code moved, not rewritten -
//   kv_dequant instantiates it and tb_kv_dequant is the proof that
//   nothing changed.
//
//====================================================================
// THE SIGN FLIP MOVES, AND THAT IS THE WHOLE SUBTLETY
//   Forward (kv_rotate) is y = H*(D*x): D first, then the butterfly.
//   So the inverse is x = D*(H*y)/N: butterfly FIRST, D after.
//   Applying D on the way in would be a different, wrong operation
//   that still looks symmetric. Mode 1's inverse is Pi^T, which is the
//   same ROM addressed {ij, oi} instead of {oi, ij}.
//
//   NO DIVIDE BY N HERE. H*H = N*I exactly, so the butterfly leaves
//   the result N times too large, and N is a power of two. Callers
//   fold that shift into scaling they were already doing:
//   kv_dequant folds it into INV_MUL/INV_SHIFT beside the level
//   rescale, and the attention path folds it into the requant tap's
//   shift_amt beside 1/l. Keeping the scale out of the datapath is
//   what lets both testbenches demand bit-exact agreement rather than
//   a tolerance - the same argument kv_rotate makes on the way in.
//
//====================================================================
// WHY run IS A LEVEL AND NOT A start PULSE
//   Both callers already own a sequencer that parks in a "rotating"
//   state until the butterfly reports done, so a level input is what
//   they have and a pulse would mean inventing a latch at each site.
//   It is also what makes this extraction provably behaviour-
//   preserving: `state == D_ROT` in the original becomes `run` here,
//   character for character, with no new edge detection to get wrong.
//
//   The read port is COMBINATIONAL, with D already applied, because
//   that is how kv_dequant consumed it - out_data fed a scale and a
//   saturate that were registered together downstream. Registering it
//   here would have put the data one cycle ahead of its own index at
//   the existing call site.
//====================================================================
module kv_irotate #(
    parameter int N        = 128,
    parameter int DW       = 32,   // element width
    parameter int ROT_MODE = 0,
    parameter ROT_SIGN_PATH = "rtl/kv_rot_sign.memh",
    parameter ROT_PI_PATH   = "rtl/kv_rot_pi.memh"
)(
    input  logic clk,
    input  logic rst_n,

    // load one element per cycle, any order the caller likes
    input  logic                 in_valid,
    input  logic [$clog2(N)-1:0] in_idx,
    input  logic [DW-1:0]        in_data,

    // hold high for the whole butterfly; drop it to reset the counters
    input  logic                 run,
    output logic                 calc_done,

    // combinational read, D already applied, still N times too large
    input  logic [$clog2(N)-1:0] rd_idx,
    output logic [DW-1:0]        rd_data
);

    localparam int AW     = $clog2(N);
    localparam int STAGES = AW;
    localparam int HALF   = N / 2;

    localparam logic [AW-1:0] ONE     = 1;
    localparam logic [AW-1:0] IJ_LAST = N - 1;

    if ((1 << AW) != N) begin : gen_n_pow2
        initial begin
            $error("kv_irotate: N (%0d) must be a power of two", N);
            $fatal(1);
        end
    end

    logic [DW-1:0] mem [N];

    generate
    if (ROT_MODE == 0) begin : gen_ihad
        //------------------------------------------------------------
        // Inverse randomized Hadamard: butterfly first, D afterwards.
        //------------------------------------------------------------
        (* rom_style = "distributed" *) logic [7:0] sign_rom [N];
        initial $readmemh(ROT_SIGN_PATH, sign_rom);

        logic [2:0]    st;
        logic [AW-2:0] pr;
        logic [AW-1:0] pr_ext, mask_lo, idx_i, idx_j;
        logic [DW-1:0] a_val, b_val, raw;

        localparam logic [AW-2:0] PR_LAST = HALF - 1;

        assign pr_ext  = {1'b0, pr};
        assign mask_lo = (ONE << st) - ONE;
        assign idx_i   = ((pr_ext >> st) << (st + 1'b1)) | (pr_ext & mask_lo);
        assign idx_j   = idx_i | (ONE << st);
        assign a_val   = mem[idx_i];
        assign b_val   = mem[idx_j];

        always_ff @(posedge clk) begin
            if (in_valid) begin
                mem[in_idx] <= in_data;
            end else if (run) begin
                mem[idx_i] <= a_val + b_val;
                mem[idx_j] <= a_val - b_val;
            end
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                st <= 3'd0;
                pr <= '0;
            end else if (!run) begin
                st <= 3'd0;
                pr <= '0;
            end else if (pr == PR_LAST) begin
                pr <= '0;
                st <= st + 3'd1;
            end else begin
                pr <= pr + 1'b1;
            end
        end

        assign calc_done = run && (st == STAGES - 1) && (pr == PR_LAST);
        // D applied here, on the way out
        assign raw     = mem[rd_idx];
        assign rd_data = sign_rom[rd_idx][0] ? (~raw + 1'b1) : raw;

    end else begin : gen_idense
        //------------------------------------------------------------
        // Inverse dense: x = Pi^T y, i.e. address the ROM transposed.
        //------------------------------------------------------------
        (* rom_style = "block" *) logic [7:0] pi_rom [N*N];
        initial $readmemh(ROT_PI_PATH, pi_rom);

        logic [DW-1:0]      ymem [N];
        logic [AW-1:0]      oi, ij, oi_r, ij_r;
        logic [7:0]         pi_q;
        logic [DW-1:0]      y_q;
        logic signed [8:0]  pi_s;
        logic signed [DW+8:0] prod;
        logic [DW-1:0]      acc;
        logic               mac_v, ctr_last, ctr_last_r;

        always_ff @(posedge clk) begin
            if (in_valid) ymem[in_idx] <= in_data;
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                oi <= '0;
                ij <= '0;
            end else if (!run) begin
                oi <= '0;
                ij <= '0;
            end else if (ij == IJ_LAST) begin
                ij <= '0;
                oi <= oi + 1'b1;
            end else begin
                ij <= ij + 1'b1;
            end
        end

        // {ij, oi} rather than {oi, ij}: that transposition IS the
        // inverse rotation.
        always_ff @(posedge clk) begin
            pi_q <= pi_rom[{ij, oi}];
            y_q  <= ymem[ij];
            oi_r <= oi;
            ij_r <= ij;
        end

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) mac_v <= 1'b0;
            else        mac_v <= run;
        end

        assign pi_s = {pi_q[7], pi_q};
        assign prod = pi_s * $signed(y_q);

        always_ff @(posedge clk) begin
            if (!mac_v)               acc <= '0;
            else if (ij_r == IJ_LAST) acc <= '0;
            else                      acc <= acc + prod[DW-1:0];
        end

        always_ff @(posedge clk) begin
            if (mac_v && (ij_r == IJ_LAST))
                mem[oi_r] <= acc + prod[DW-1:0];
        end

        assign ctr_last = run && (oi == IJ_LAST) && (ij == IJ_LAST);

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) ctr_last_r <= 1'b0;
            else        ctr_last_r <= ctr_last;
        end

        assign calc_done = ctr_last_r;
        assign rd_data   = mem[rd_idx];
    end
    endgenerate

endmodule
