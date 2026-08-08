`timescale 1ns / 1ps
//====================================================================
// tb_recip - 1/x as mantissa and shift.
//
//   THE GOLDEN IS VERILOG'S OWN DIVIDE, not a second shift-subtract
//   loop. Checking a restoring divider against a hand-written model of
//   restoring division proves the two share a bug, not that either is
//   right; `/` on a 64-bit reg is a genuinely independent path.
//
//   R1  the defining property, on every case: mant is EXACTLY
//       floor(2**shift / x), saturated at 2**MW-1.
//   R2  normalization: mant's top bit is always set, so shift is the
//       unique one R1 admits and the mantissa never wastes bits. R1
//       alone would pass for shift too small.
//   R3  the real numbers - both l values from the attention
//       end-to-end run - reconstruct 1/l to better than 2**-(MW-1)
//       relative. This is the number that actually has to be right.
//   R4  the saturating edge, x an exact power of two, is the ONE
//       input where mant is not exact. Checked deliberately rather
//       than avoided.
//   R5  x = 0 raises zero rather than returning a plausible number.
//   R6  back-to-back calls, because this runs once per head and will
//       be invoked 16 times in a row.
//====================================================================
module tb_recip;

    localparam int XW = 48;
    localparam int MW = 15;
    localparam int SW = 6;

    logic clk = 1'b0, rst_n = 1'b0;
    always #5 clk = ~clk;

    localparam [63:0] MANT_MAX = (1 << MW) - 1;

    integer errors = 0;

    logic          go, busy, done, zero;
    logic [XW-1:0] x;
    logic [MW-1:0] mant;
    logic [SW-1:0] shift;

    recip #(.XW(XW), .MW(MW), .SW(SW)) dut (
        .clk(clk), .rst_n(rst_n), .go(go), .x(x),
        .busy(busy), .done(done), .mant(mant), .shift(shift), .zero(zero));

    // 64 bits is enough: shift maxes at XW-1+MW = 62.
    logic [63:0] num, gold;
    real         approx, truth, relerr;

    // $itor truncates to 32 bits here, which silently turns a 48-bit
    // operand into a different number - R3 "failed" on values R1 had
    // just passed until this was split. Two 24-bit halves, both
    // unambiguously positive.
    function automatic real u48(input [XW-1:0] v);
        u48 = $itor(v[47:24]) * 16777216.0 + $itor(v[23:0]);
    endfunction

    task automatic call(input [XW-1:0] xv);
        begin
            @(negedge clk);
            x  = xv;
            go = 1'b1;
            @(negedge clk);
            go = 1'b0;
            @(posedge done);
            @(negedge clk);
        end
    endtask

    // R1 + R2 on one input.
    task automatic check(input [XW-1:0] xv, input string tag);
        begin
            call(xv);
            num  = 64'd1 << shift;
            gold = num / {16'd0, xv};
            if (gold > MANT_MAX) gold = MANT_MAX;
            if ({49'd0, mant} !== gold) begin
                errors = errors + 1;
                $display("Error: R1 %s x=%0d shift=%0d mant=%0d exp=%0d",
                         tag, xv, shift, mant, gold);
            end
            if (mant[MW-1] !== 1'b1) begin
                errors = errors + 1;
                $display("Error: R2 %s x=%0d mant=%0d not normalized",
                         tag, xv, mant);
            end
            if (zero !== 1'b0) begin
                errors = errors + 1;
                $display("Error: R1 %s x=%0d zero flag set", tag, xv);
            end
        end
    endtask

    integer i;
    logic [XW-1:0] xv;
    integer n1, n3, n4;

    initial begin
        go = 0; x = '0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        $display("=== recip: 1/x as mantissa + shift, XW=%0d MW=%0d ===", XW, MW);

        // ---- R1/R2: a spread that reaches both ends of the range ----
        n1 = errors;
        check(48'd1,                "min");
        check(48'd2,                "two");
        check(48'd3,                "three");
        check(48'd7,                "seven");
        check(48'd255,              "255");
        check(48'd1000,             "1000");
        check(48'h0000_0001_0000,   "2^16");
        check(48'h0000_0002_AAAB,   "175275");
        check(48'h0000_4000_0000,   "2^30");
        check(48'hFFFF_FFFF_FFFF,   "max");
        check(48'h8000_0000_0000,   "2^47");
        check(48'h7FFF_FFFF_FFFF,   "2^47-1");
        // pseudo-random sweep, deterministic
        for (i = 0; i < 64; i = i + 1) begin
            // deterministic spread; the multiplier is prime and the
            // XOR keeps the high half from repeating in step
            xv = (48'h9E3779B97F4A * (i + 1)) ^ (48'd1 << (i % 47));
            if (xv == 48'd0) xv = 48'd1;
            check(xv, "sweep");
        end
        $display("  R1/R2 %0d inputs: mant == floor(2^shift/x) and normalized, %0d errors",
                 12 + 64, errors - n1);

        // ---- R3: the real l values from the attention run ----------
        n3 = errors;
        // head 0 and head 1 of tb_attn_e2e / tb_attn_kv_top
        for (i = 0; i < 2; i = i + 1) begin
            xv = (i == 0) ? 48'd31410416640 : 48'd10157184644;
            call(xv);
            truth  = 1.0 / u48(xv);
            approx = $itor(mant) / $pow(2.0, $itor(shift));
            relerr = (approx - truth) / truth;
            if (relerr < 0.0) relerr = -relerr;
            if (relerr > $pow(2.0, -$itor(MW - 1))) begin
                errors = errors + 1;
                $display("Error: R3 l=%0d relerr %e exceeds 2^-%0d",
                         xv, relerr, MW - 1);
            end
            $display("  R3 l=%0d -> mant %0d shift %0d, rel err %e",
                     xv, mant, shift, relerr);
        end
        if (errors == n3) $display("  R3 both real normalizers within 2^-%0d", MW-1);

        // ---- R4: the saturating edge, on purpose -------------------
        n4 = errors;
        for (i = 0; i < XW; i = i + 1) begin
            xv = 48'd1 << i;
            call(xv);
            // exact answer would be 2^MW; the module saturates one below
            if (mant !== {MW{1'b1}}) begin
                errors = errors + 1;
                $display("Error: R4 x=2^%0d mant %0d expected %0d",
                         i, mant, (1 << MW) - 1);
            end
            if (shift !== (i + MW)) begin
                errors = errors + 1;
                $display("Error: R4 x=2^%0d shift %0d expected %0d",
                         i, shift, i + MW);
            end
        end
        $display("  R4 all %0d powers of two saturate to %0d with shift = e+%0d, %0d errors",
                 XW, (1 << MW) - 1, MW, errors - n4);

        // ---- R5: zero is loud ---------------------------------------
        call(48'd0);
        if (zero !== 1'b1) begin
            errors = errors + 1;
            $display("Error: R5 x=0 did not raise zero");
        end
        // and it clears on the next good call
        call(48'd12345);
        if (zero !== 1'b0) begin
            errors = errors + 1;
            $display("Error: R5 zero stayed set after a valid call");
        end
        $display("  R5 x=0 raises zero, cleared by the next call");

        // ---- R6: back to back, 16 in a row (one per Q head) --------
        n1 = errors;
        for (i = 0; i < 16; i = i + 1)
            check(48'd31410416640 + i, "b2b");
        $display("  R6 16 consecutive calls, %0d errors", errors - n1);

        if (errors == 0) $display("ALL TESTS PASSED");
        else             $display("FAILED with %0d errors", errors);
        $finish;
    end

    initial begin
        #10000000;
        $display("Error: timeout");
        $display("FAILED");
        $finish;
    end

endmodule
