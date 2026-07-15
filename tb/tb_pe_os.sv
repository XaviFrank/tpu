`timescale 1ns / 1ps
//====================================================================
// tb_pe_os: self-checking testbench for the output-stationary PE
//   T1: dot product [1,2,3]·[4,5,6] = 32
//   T2: gating — garbage with valid low must not disturb acc
//   T3: clear zeroes the accumulator
//   T4: signed dot product [-3,5]·[7,-2] = -31
//   T5: clear beats a straggler product (priority check)
//   Continuous: a/b/valid forwarding is exactly one cycle
//====================================================================
module tb_pe_os;

    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;

    // DUT I/O
    logic clk = 1'b0;
    logic rst_n, clear, valid_in;
    logic signed [DATA_WIDTH-1:0] a_in, b_in;
    logic                         valid_out;
    logic signed [DATA_WIDTH-1:0] a_out, b_out;
    logic signed [ACC_WIDTH-1:0]  acc_out;
    logic                         acc_valid;
    logic                         drain;
    logic signed [ACC_WIDTH-1:0]  acc_in_above;

    int errors = 0;

    pe_os #(
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear     (clear),
        .valid_in  (valid_in),
        .a_in      (a_in),
        .b_in      (b_in),
        .valid_out (valid_out),
        .a_out     (a_out),
        .b_out     (b_out),
        .acc_out   (acc_out),
        .acc_valid (acc_valid)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    //----------------------------------------------------------------
    // Drive one input beat. Driving on the NEGEDGE means signals are
    // stable long before the posedge samples them - no races.
    //----------------------------------------------------------------
    task automatic drive(input logic v,
                         input logic signed [DATA_WIDTH-1:0] a,
                         input logic signed [DATA_WIDTH-1:0] b);
        @(negedge clk);
        valid_in = v;
        a_in     = a;
        b_in     = b;
    endtask

    //----------------------------------------------------------------
    // Compare acc_out against expected, at the moment of the call
    //----------------------------------------------------------------
    task automatic check_acc(input logic signed [ACC_WIDTH-1:0] exp,
                             input string name);
        if (acc_out !== exp) begin
            errors++;
            $error("[%s] acc_out = %0d, expected %0d", name, acc_out, exp);
        end else begin
            $display("PASS [%s] acc_out = %0d", name, acc_out);
        end
    endtask

    //----------------------------------------------------------------
    // Continuous forwarding checker: a_out/b_out/valid_out must equal
    // last cycle's inputs, every single cycle. Runs in the background
    // for the whole test - this is your systolic-timing guarantee.
    //----------------------------------------------------------------
    logic signed [DATA_WIDTH-1:0] a_exp, b_exp;
    logic v_exp;
    logic chk_en = 1'b0;

    always @(posedge clk) begin
        if (chk_en) begin
            if (a_out   !== a_exp) begin errors++; $error("a_out=%0d, expected %0d (1-cycle hop broken)", a_out, a_exp); end
            if (b_out   !== b_exp) begin errors++; $error("b_out=%0d, expected %0d (1-cycle hop broken)", b_out, b_exp); end
            if (valid_out !== v_exp) begin errors++; $error("valid_out=%b, expected %b", valid_out, v_exp); end
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
        rst_n = 0; clear = 0; valid_in = 0; a_in = '0; b_in = '0; drain = 0; acc_in_above = '0;
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        clear = 1;
        @(negedge clk);
        clear = 0;
        @(negedge clk);
        chk_en = 1;

        //--- T1: [1,2,3] . [4,5,6] = 4 + 10 + 18 = 32 ---------------
        drive(1, 8'sd1, 8'sd4);
        drive(1, 8'sd2, 8'sd5);
        drive(1, 8'sd3, 8'sd6);
        drive(0, 8'sd0, 8'sd0);      // stop feeding
        @(negedge clk);              // let the pipeline drain (2 stages)
        check_acc(32, "T1 dot product");

        //--- T2: garbage with valid low must not move the acc -------
        drive(0, 8'sh7F, 8'sh7F);    // 127*127 on the wires, valid=0
        drive(0, -8'sd128, 8'sd99);
        @(negedge clk);
        check_acc(32, "T2 valid gating");

        //--- T3: clear zeroes the accumulator ------------------------
        @(negedge clk) clear = 1;
        @(negedge clk) clear = 0;
        check_acc(0, "T3 clear");

        //--- T4: signed: (-3)(7) + (5)(-2) = -31 ---------------------
        drive(1, -8'sd3, 8'sd7);
        drive(1,  8'sd5, -8'sd2);
        drive(0,  8'sd0, 8'sd0);
        @(negedge clk);
        check_acc(-31, "T4 signed dot product");

        // reset acc for T5
        @(negedge clk) clear = 1;
        @(negedge clk) clear = 0;

        //--- T5: clear vs. straggler product -------------------------
        // Feed one valid beat, then assert clear on the very next
        // cycle so clear and valid_pipe collide. Clear must win:
        // acc stays 0, and 81 must NOT leak in afterward.
        drive(1, 8'sd9, 8'sd9);      // product 81 enters the pipe
        @(negedge clk) begin valid_in = 0; clear = 1; end
        @(negedge clk) clear = 0;
        @(negedge clk);
        check_acc(0, "T5 clear priority over straggler");

        //--- summary --------------------------------------------------
        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule