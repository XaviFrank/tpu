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

    // PIPE_STAGES equivalence: the same golden vectors are replayed
    // against the pipelined variants, so raising PIPE_STAGES at one
    // call site cannot change the arithmetic. dut is the PIPE_STAGES=0
    // instance the original TB already certified - untouched, and it
    // is what OS's frozen tap and ws_tiling_engine's tap keep using.
    logic clk = 1'b0;
    logic signed [7:0] out_p1, out_p2;

    always #5 clk = ~clk;

    requant #(
        .ACC_WIDTH(32), .OUT_WIDTH(8), .SCALE_WIDTH(16), .SHIFT_WIDTH(5)
    ) dut (
        .clk        (clk),
        .acc_in     (acc_in),
        .relu_en    (relu_en),
        .scale_mult (scale_mult),
        .shift_amt  (shift_amt),
        .out        (out)
    );

    requant #(
        .ACC_WIDTH(32), .OUT_WIDTH(8), .SCALE_WIDTH(16), .SHIFT_WIDTH(5),
        .PIPE_STAGES(1)
    ) dut_p1 (
        .clk        (clk),
        .acc_in     (acc_in),
        .relu_en    (relu_en),
        .scale_mult (scale_mult),
        .shift_amt  (shift_amt),
        .out        (out_p1)
    );

    requant #(
        .ACC_WIDTH(32), .OUT_WIDTH(8), .SCALE_WIDTH(16), .SHIFT_WIDTH(5),
        .PIPE_STAGES(2)
    ) dut_p2 (
        .clk        (clk),
        .acc_in     (acc_in),
        .relu_en    (relu_en),
        .scale_mult (scale_mult),
        .shift_amt  (shift_amt),
        .out        (out_p2)
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

        //--- P1: PIPE_STAGES 1 and 2 compute the SAME math -----------
        // Inputs held stable, so both pipes have settled: this is a
        // pure arithmetic-equivalence check against the same
        // independent Python golden the combinational core is held to.
        begin
            int bad;
            bad = 0;
            for (int i = 0; i < NVEC_MAX; i++) begin
                if ($isunknown(vecs[i])) continue;
                @(negedge clk) begin
                    acc_in     = vecs[i][63:32];
                    scale_mult = vecs[i][31:16];
                    shift_amt  = vecs[i][15:11];
                    relu_en    = vecs[i][10];
                    exp_out    = vecs[i][7:0];
                end
                repeat (3) @(negedge clk);          // let both pipes fill
                if (out_p1 !== exp_out) begin
                    bad++; errors++;
                    $error("[P1 pipe1 vec %0d] out=%0d, expected %0d", i, out_p1, exp_out);
                end
                if (out_p2 !== exp_out) begin
                    bad++; errors++;
                    $error("[P1 pipe2 vec %0d] out=%0d, expected %0d", i, out_p2, exp_out);
                end
            end
            if (bad == 0)
                $display("PASS [P1] PIPE_STAGES=1 and =2 match the golden on all %0d vectors",
                         n_run);
        end

        //--- P2: latency is EXACTLY PIPE_STAGES cycles ---------------
        // Equivalence alone would also pass if a "pipelined" variant
        // were secretly combinational, so drive a new value every
        // cycle and check each output appears at its own offset.
        begin
            logic signed [7:0] seq_exp [4];
            int bad;
            bad = 0;
            scale_mult = 16'sd1;                    // identity: out = clamp(acc)
            shift_amt  = 5'd0;
            relu_en    = 1'b0;
            seq_exp[0] = 8'sd10; seq_exp[1] = 8'sd20;
            seq_exp[2] = 8'sd30; seq_exp[3] = 8'sd40;

            @(negedge clk) acc_in = 32'sd10;
            @(negedge clk) acc_in = 32'sd20;
            // after one clk, pipe1 shows vector 0 while pipe2 shows nothing yet
            if (out_p1 !== seq_exp[0]) begin
                bad++; errors++;
                $error("[P2] pipe1 latency: out=%0d after 1 cycle, expected %0d",
                       out_p1, seq_exp[0]);
            end
            @(negedge clk) acc_in = 32'sd30;
            if (out_p1 !== seq_exp[1]) begin
                bad++; errors++;
                $error("[P2] pipe1 latency: out=%0d, expected %0d", out_p1, seq_exp[1]);
            end
            if (out_p2 !== seq_exp[0]) begin
                bad++; errors++;
                $error("[P2] pipe2 latency: out=%0d after 2 cycles, expected %0d",
                       out_p2, seq_exp[0]);
            end
            @(negedge clk) acc_in = 32'sd40;
            if (out_p2 !== seq_exp[1]) begin
                bad++; errors++;
                $error("[P2] pipe2 latency: out=%0d, expected %0d", out_p2, seq_exp[1]);
            end
            if (bad == 0)
                $display("PASS [P2] latency is exactly 1 and 2 cycles, and both stream back-to-back");
        end

        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
