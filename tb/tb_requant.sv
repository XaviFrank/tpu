`timescale 1ns / 1ps
//====================================================================
// tb_requant: replays Python-generated golden vectors against the
// combinational requant core.
//   Golden model lives in tb/gen_requant_vectors.py (independent
//   Python implementation of the ReLU/scale/round/shift/saturate
//   math - NOT a SystemVerilog re-derivation); this TB only unpacks
//   bits and compares with !==. Vector word layout documented in
//   the generator. Directed cases cover: mid-range, +/-127/128
//   exact boundaries and first-clipping values, ReLU on/off on
//   negatives, round-half-up on +/-0.5, shift_amt=0 (no-bias path),
//   extreme acc/shift corners; plus a seeded random sweep.
//====================================================================
module tb_requant;

    localparam int NVEC_MAX = 256;

    // DUT I/O
    logic signed [31:0] acc_in;
    logic               relu_en;
    logic signed [15:0] scale_mult;
    logic [4:0]         shift_amt;
    logic signed [7:0]  out;

    logic [63:0] vecs [NVEC_MAX];
    logic signed [7:0] exp_out;

    int errors = 0;
    int n_run  = 0;

    requant #(
        .ACC_WIDTH(32), .OUT_WIDTH(8), .SCALE_WIDTH(16), .SHIFT_WIDTH(5)
    ) dut (
        .acc_in     (acc_in),
        .relu_en    (relu_en),
        .scale_mult (scale_mult),
        .shift_amt  (shift_amt),
        .out        (out)
    );

    initial begin
        for (int i = 0; i < NVEC_MAX; i++)
            vecs[i] = 64'hx;
        $readmemh("tb/requant_vectors.memh", vecs);

        for (int i = 0; i < NVEC_MAX; i++) begin
            if ($isunknown(vecs[i]))
                continue;                      // past the last vector

            acc_in     = vecs[i][63:32];
            scale_mult = vecs[i][31:16];
            shift_amt  = vecs[i][15:11];
            relu_en    = vecs[i][10];
            exp_out    = vecs[i][7:0];
            #1;                                // combinational settle

            if (out !== exp_out) begin
                errors++;
                $error("[vec %0d] acc=%0d scale=%0d shift=%0d relu=%b: out=%0d, expected %0d",
                       i, acc_in, scale_mult, shift_amt, relu_en, out, exp_out);
            end
            n_run++;
        end

        if (n_run == 0) begin
            errors++;
            $error("no vectors loaded - tb/requant_vectors.memh missing?");
        end else begin
            $display("ran %0d golden vectors, %0d error(s)", n_run, errors);
        end

        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
