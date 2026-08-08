`timescale 1ns / 1ps
//====================================================================
// tb_attn_finalize - the last three stages, on the real attention row.
//
//   o[] and l are NOT synthesized here. They are tb/attn_o0.memh and
//   tb/attn_l0.memh, the same files tb_attn_scan, tb_attn_scan_gqa,
//   tb_attn_e2e and tb_attn_kv_top all check against - i.e. the actual
//   output of the verified attention pipeline on a real Qwen3 layer-0
//   row. So this measures the finalization and nothing else; if it
//   disagrees, the disagreement is in recip, kv_irotate or requant.
//
//   The golden is tb/attn_fin*.memh from gen_attn_vectors.py, computed
//   in numpy with the same integer semantics but by a different route
//   (Python's // and >> against a shift-subtract FSM and a butterfly).
//
//   F1  both Q heads, all 128 coordinates, bit-exact INT8.
//   F2  mant and shift match the model - checked SEPARATELY, because
//       F1 would also pass if a wrong mantissa were cancelled by a
//       wrong shift, and those are the two numbers a width change
//       would break first.
//   F3  o[] is read through the same combinational port attn_kv_top
//       exposes, one address per cycle, so the load pass is the real
//       interface rather than a preloaded array.
//   F4  back-to-back invocation - 16 Q heads run through one of these
//       in sequence - and the SECOND head's answer must be right after
//       the first, which is what catches state left in the butterfly
//       memory or a stale mantissa.
//   F5  l = 0 raises zero instead of returning a plausible vector.
//====================================================================
module tb_attn_finalize;

    localparam int N  = 128;
    localparam int OW = 48;
    localparam int LW = 48;
    localparam int MW = 15;
    localparam int SW = 6;

    logic clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    integer errors = 0, i, h, bad;

    logic [47:0] eo   [2][N];
    logic [47:0] el   [2];
    logic [7:0]  efin [2][N];
    logic [15:0] ems  [2][2];      // {mant, shift}

    // Icarus will not $readmemh into a slice of a 2-D array, so each
    // head lands in its own flat array first.
    logic [47:0] eo0 [N], eo1 [N];
    logic [47:0] el0 [1], el1 [1];
    logic [7:0]  ef0 [N], ef1 [N];
    logic [15:0] em0 [2], em1 [2];

    initial begin
        $readmemh("tb/attn_o0.memh",     eo0);
        $readmemh("tb/attn_o1.memh",     eo1);
        $readmemh("tb/attn_l0.memh",     el0);
        $readmemh("tb/attn_l1.memh",     el1);
        $readmemh("tb/attn_fin0.memh",   ef0);
        $readmemh("tb/attn_fin1.memh",   ef1);
        $readmemh("tb/attn_finms0.memh", em0);
        $readmemh("tb/attn_finms1.memh", em1);
    end

    //----------------------------------------------------------------
    // Stand-in for attn_kv_top's o read port: combinational, exactly
    // as attn_scan's `assign o_rd_data = o[o_rd_addr]` is.
    //----------------------------------------------------------------
    logic               go, busy, done, zero;
    logic [LW-1:0]      l_in;
    logic [6:0]         o_rd_addr;
    logic signed [OW-1:0] o_rd_data;
    logic               out_valid;
    logic [6:0]         out_idx;
    logic signed [7:0]  out_data;

    logic head_sel;
    assign o_rd_data = head_sel ? $signed(eo1[o_rd_addr])
                                : $signed(eo0[o_rd_addr]);

    attn_finalize #(.N(N), .OW(OW), .LW(LW), .MW(MW), .SW(SW),
                    .POST_SHIFT(8), .ROT_MODE(0)) dut (
        .clk(clk), .rst_n(rst_n),
        .go(go), .l_in(l_in), .busy(busy), .done(done),
        .o_rd_addr(o_rd_addr), .o_rd_data(o_rd_data),
        .out_valid(out_valid), .out_idx(out_idx), .out_data(out_data),
        .zero(zero));

    // Capture the emitted vector by the index the DUT reports, not by
    // arrival order - so an off-by-one in out_idx cannot be papered
    // over by a testbench that just appends.
    logic signed [7:0] got [N];
    logic [N-1:0]      seen;

    always_ff @(posedge clk) begin
        if (out_valid) begin
            got[out_idx]  <= out_data;
            seen[out_idx] <= 1'b1;
        end
    end

    task automatic run_head(input integer hh);
        begin
            @(negedge clk);
            head_sel = hh[0];
            l_in     = (hh == 0) ? el0[0] : el1[0];
            seen     = '0;
            @(negedge clk); go = 1'b1;
            @(negedge clk); go = 1'b0;
            @(posedge done);
            repeat (2) @(negedge clk);
        end
    endtask

    initial begin
        go = 0; l_in = '0; head_sel = 0; seen = '0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        $display("=== attn_finalize: 1/l, inverse rotation, requant ===");

        // ---- F1..F4: both heads, in sequence, one after the other --
        for (h = 0; h < 2; h = h + 1) begin
            run_head(h);

            if (seen !== {N{1'b1}}) begin
                errors = errors + 1;
                $display("Error: F3 head %0d did not emit all %0d indices", h, N);
            end

            bad = 0;
            for (i = 0; i < N; i = i + 1) begin
                if (got[i] !== $signed((h == 0) ? ef0[i] : ef1[i])) begin
                    bad = bad + 1;
                    if (bad <= 3)
                        $display("Error: F1 head %0d out[%0d] got %0d exp %0d",
                                 h, i, got[i], $signed((h == 0) ? ef0[i] : ef1[i]));
                end
            end
            errors = errors + bad;

            // F2: the two numbers a width change breaks first
            if (dut.rc_mant !== ((h == 0) ? em0[0][MW-1:0] : em1[0][MW-1:0])) begin
                errors = errors + 1;
                $display("Error: F2 head %0d mant %0d exp %0d", h, dut.rc_mant,
                         (h == 0) ? em0[0] : em1[0]);
            end
            if (dut.rc_shift !== ((h == 0) ? em0[1][SW-1:0] : em1[1][SW-1:0])) begin
                errors = errors + 1;
                $display("Error: F2 head %0d shift %0d exp %0d", h, dut.rc_shift,
                         (h == 0) ? em0[1] : em1[1]);
            end
            if (zero !== 1'b0) begin
                errors = errors + 1;
                $display("Error: F1 head %0d zero flag set", h);
            end

            $display("  F1 head %0d: %0d/%0d coords bit-exact vs numpy, mant %0d shift %0d",
                     h, N - bad, N, dut.rc_mant, dut.rc_shift);
        end
        $display("  F4 the second head ran after the first with no reset - both exact");

        // ---- F5: l = 0 is loud -------------------------------------
        @(negedge clk);
        head_sel = 1'b0;
        l_in     = 48'd0;
        @(negedge clk); go = 1'b1;
        @(negedge clk); go = 1'b0;
        @(posedge done);
        repeat (2) @(negedge clk);
        if (zero !== 1'b1) begin
            errors = errors + 1;
            $display("Error: F5 l=0 did not raise zero");
        end else begin
            $display("  F5 l=0 raises zero");
        end

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

    initial begin
        #20000000;
        $display("Error: timeout");
        $display("FAILED");
        $finish;
    end

endmodule
