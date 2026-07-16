`timescale 1ns / 1ps
//====================================================================
// tb_ws_pe: self-checking TB for the weight-stationary PE
//   T1a: LOAD passthrough + w_capture latches the word in sum_reg
//   T1b: w_reg holds across STREAM-mode cycles (b/a wiggling, valid
//        low - nothing may disturb the resident weight)
//   T1c: clean re-entry into LOAD for a second weight (negative,
//        checks the low-byte capture keeps the sign)
//   T2 : K-length STREAM accumulate: b_out(k) = psum_in + a*w with
//        the 2-stage pairing (product of a from cycle c adds to the
//        b_in of cycle c+1); sum_valid tracks it exactly
//   T2b: valid gating - garbage a with valid low contributes 0,
//        the psum passes through untouched
//   T3 : reload a new weight after streaming, stream again - no
//        corruption from the first tile (sum path has no persistent
//        state; this proves it)
//   Continuous: a/valid 1-cycle hop always; b passthrough +
//   sum_valid==0 whenever no valid product is in flight (chk_pass)
//====================================================================
module tb_ws_pe;

    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;

    // DUT I/O
    logic clk = 1'b0;
    logic rst_n, load_mode, w_capture, valid_in;
    logic signed [DATA_WIDTH-1:0] a_in;
    logic signed [ACC_WIDTH-1:0]  b_in;
    logic                         valid_out;
    logic signed [DATA_WIDTH-1:0] a_out;
    logic signed [ACC_WIDTH-1:0]  b_out;
    logic                         sum_valid;

    int errors = 0;

    ws_pe #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .load_mode (load_mode),
        .w_capture (w_capture),
        .valid_in  (valid_in),
        .a_in      (a_in),
        .b_in      (b_in),
        .valid_out (valid_out),
        .a_out     (a_out),
        .b_out     (b_out),
        .sum_valid (sum_valid)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    //----------------------------------------------------------------
    // Shift one weight word in and capture it (the controller's
    // LOAD_WEIGHTS protocol, single-PE version)
    //----------------------------------------------------------------
    task automatic load_weight(input int w);
        @(negedge clk) begin load_mode = 1; b_in = w; end
        @(negedge clk) begin b_in = '0; w_capture = 1; end
        @(negedge clk) begin w_capture = 0; load_mode = 0; end
    endtask

    //----------------------------------------------------------------
    // Check the resident weight via hierarchical peek (unit TB only).
    // Golden value handling: raw bits out, manual two's complement -
    // no $signed (known Icarus issue).
    //----------------------------------------------------------------
    task automatic check_w(input int exp, input string name);
        int got;
        got = {24'b0, dut.w_reg};
        if (got >= 128) got = got - 256;   // manual two's complement
        if (got !== exp) begin
            errors++;
            $error("[%s] w_reg = %0d, expected %0d", name, got, exp);
        end else begin
            $display("PASS [%s] w_reg = %0d", name, got);
        end
    endtask

    //----------------------------------------------------------------
    // Check the psum output and its valid, at the moment of the call
    //----------------------------------------------------------------
    task automatic check_out(input int exp, input logic v,
                             input string name);
        if (b_out !== exp) begin
            errors++;
            $error("[%s] b_out = %0d, expected %0d", name, b_out, exp);
        end else if (sum_valid !== v) begin
            errors++;
            $error("[%s] sum_valid = %b, expected %b", name, sum_valid, v);
        end else begin
            $display("PASS [%s] b_out = %0d (sum_valid=%b)", name, b_out, sum_valid);
        end
    endtask

    //----------------------------------------------------------------
    // Continuous checkers. Hops (a, valid) hold ALWAYS; passthrough
    // (b_out == last b_in) and sum_valid==0 hold only while no valid
    // product is in flight - gated by chk_pass.
    //----------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] a_exp;
    logic signed [ACC_WIDTH-1:0]  b_exp;
    logic v_exp;
    logic chk_en   = 1'b0;
    logic chk_pass = 1'b0;

    always @(posedge clk) begin
        if (chk_en) begin
            if (a_out     !== a_exp) begin errors++; $error("a_out=%0d, expected %0d (1-cycle hop broken)", a_out, a_exp); end
            if (valid_out !== v_exp) begin errors++; $error("valid_out=%b, expected %b", valid_out, v_exp); end
        end
        if (chk_en && chk_pass) begin
            if (b_out     !== b_exp) begin errors++; $error("b_out=%0d, expected %0d (passthrough broken)", b_out, b_exp); end
            if (sum_valid !== 1'b0)  begin errors++; $error("sum_valid=%b, must stay 0 while idle", sum_valid); end
        end
        a_exp <= a_in;
        b_exp <= b_in;
        v_exp <= valid_in;
    end

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    initial begin
        // init & reset
        rst_n = 0; load_mode = 0; w_capture = 0; valid_in = 0;
        a_in = '0; b_in = '0;
        repeat (3) @(negedge clk);
        rst_n = 1;

        //--- T1a: shift 42 in, capture it -----------------------------
        @(negedge clk) begin load_mode = 1; b_in = 32'sd42; end
        chk_en = 1; chk_pass = 1;                // b_exp now primed
        @(negedge clk) begin b_in = 32'sd99; w_capture = 1; end
        // posedge here: w_reg <= sum_reg (=42), sum_reg <= 99
        @(negedge clk) begin w_capture = 0; load_mode = 0; end
        check_w(42, "T1a capture");

        //--- T1b: hold across STREAM-mode cycles, valid low -----------
        // b and a wiggle hard; w_capture stays low; w_reg must not move
        for (int i = 0; i < 8; i++) begin
            @(negedge clk) begin
                b_in = 32'sd1000 + i;
                a_in = 8'sh7F;
            end
            check_w(42, "T1b hold");
        end

        //--- T1c: second load, negative weight ------------------------
        @(negedge clk) begin load_mode = 1; b_in = -32'sd7; end
        @(negedge clk) begin b_in = 32'sd0; w_capture = 1; end
        @(negedge clk) begin w_capture = 0; load_mode = 0; end
        check_w(-7, "T1c negative recapture");

        //--- T2: K=4 STREAM accumulate, w = 3 -------------------------
        // Pairing contract (2-stage MAC): the product of the a driven
        // on cycle c adds to the b_in driven on cycle c+1; the result
        // is visible on b_out during cycle c+2, sum_valid high.
        //   a = [1,-2,5,7], upstream psums p = [10,20,-30,40]
        //   results: 10+1*3=13, 20-2*3=14, -30+5*3=-15, 40+7*3=61
        load_weight(3);
        check_w(3, "T2 weight");
        chk_pass = 0;                            // real sums now in flight

        @(negedge clk) begin valid_in = 1; a_in =  8'sd1; b_in = 32'sd0;   end
        @(negedge clk) begin valid_in = 1; a_in = -8'sd2; b_in = 32'sd10;  end
        @(negedge clk) begin valid_in = 1; a_in =  8'sd5; b_in = 32'sd20;  end
        check_out(13, 1'b1, "T2 beat0");         // 10 + 1*3
        @(negedge clk) begin valid_in = 1; a_in =  8'sd7; b_in = -32'sd30; end
        check_out(14, 1'b1, "T2 beat1");         // 20 + (-2)*3
        @(negedge clk) begin valid_in = 0; a_in =  8'sd0; b_in = 32'sd40;  end
        check_out(-15, 1'b1, "T2 beat2");        // -30 + 5*3
        @(negedge clk) begin b_in = 32'sd0; end
        check_out(61, 1'b1, "T2 beat3");         // 40 + 7*3
        @(negedge clk);
        if (sum_valid !== 1'b0) begin
            errors++; $error("[T2 tail] sum_valid=%b, expected 0 after stream ends", sum_valid);
        end else $display("PASS [T2 tail] sum_valid low after stream");

        //--- T2b: valid gating mid-idle -------------------------------
        // garbage a with valid low: psum must pass through untouched
        @(negedge clk) begin valid_in = 0; a_in = 8'sh7F; b_in = 32'sd555; end
        @(negedge clk) begin a_in = -8'sd128; b_in = 32'sd0; end
        check_out(555, 1'b0, "T2b gating passthrough");
        chk_pass = 1;                            // idle again: invariants back on

        //--- T3: reload w = -5 after streaming, stream again ----------
        load_weight(-5);
        check_w(-5, "T3 reload");
        chk_pass = 0;
        @(negedge clk) begin valid_in = 1; a_in = 8'sd4; b_in = 32'sd0;   end
        @(negedge clk) begin valid_in = 0; a_in = 8'sd0; b_in = 32'sd100; end
        @(negedge clk) begin b_in = 32'sd0; end
        check_out(80, 1'b1, "T3 stream on new weight");  // 100 + 4*(-5)

        // a few more quiet cycles to let the continuous checkers run
        @(negedge clk);
        chk_pass = 1;
        repeat (3) @(negedge clk);

        //--- summary ---------------------------------------------------
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
