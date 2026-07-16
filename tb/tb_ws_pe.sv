`timescale 1ns / 1ps
//====================================================================
// tb_ws_pe: self-checking TB for the weight-stationary PE
//   T1a: LOAD passthrough + w_capture latches the word in sum_reg
//   T1b: w_reg holds across STREAM-mode cycles (b/a wiggling, valid
//        low - nothing may disturb the resident weight)
//   T1c: clean re-entry into LOAD for a second weight (negative,
//        checks the low-byte capture keeps the sign)
//   Continuous: a/valid 1-cycle hop; b passthrough (holds whenever
//   no valid product is in flight, which is all of T1); sum_valid
//   must stay 0 with valid never asserted
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
    // Continuous checkers: 1-cycle hops and passthrough. b_out must
    // equal last cycle's b_in for ALL of T1 (LOAD passes through
    // unconditionally; STREAM with valid low passes through too).
    // sum_valid must never rise: valid_in is never asserted here.
    //----------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] a_exp;
    logic signed [ACC_WIDTH-1:0]  b_exp;
    logic v_exp;
    logic chk_en = 1'b0;

    always @(posedge clk) begin
        if (chk_en) begin
            if (a_out     !== a_exp) begin errors++; $error("a_out=%0d, expected %0d (1-cycle hop broken)", a_out, a_exp); end
            if (b_out     !== b_exp) begin errors++; $error("b_out=%0d, expected %0d (passthrough broken)", b_out, b_exp); end
            if (valid_out !== v_exp) begin errors++; $error("valid_out=%b, expected %b", valid_out, v_exp); end
            if (sum_valid !== 1'b0)  begin errors++; $error("sum_valid=%b, must stay 0 with valid_in never asserted", sum_valid); end
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
        chk_en = 1;                              // b_exp now primed
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

        // a few more quiet cycles to let the continuous checkers run
        repeat (3) @(negedge clk);

        //--- summary ---------------------------------------------------
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
