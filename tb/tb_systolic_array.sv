`timescale 1ns / 1ps
//====================================================================
// tb_systolic_array: verify 4x4 OS array against a golden matmul
//   YOUR JOB: implement drive_skewed() - the parallelogram
//====================================================================
module tb_systolic_array;

    localparam int N          = 4;
    localparam int K          = 4;      // inner dimension (A is NxK, B is KxN)
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;

    logic clk = 0;
    logic rst_n, clear;
    logic                         valid_in [N];
    logic signed [DATA_WIDTH-1:0] a_in     [N];
    logic signed [DATA_WIDTH-1:0] b_in     [N];
    logic signed [ACC_WIDTH-1:0]  c_out    [N][N];
    logic                         acc_valid[N][N];
    logic                        drain;
    logic signed [ACC_WIDTH-1:0] drain_out [N];

    int errors = 0;

    // Test matrices and golden result
    logic signed [DATA_WIDTH-1:0] A [N][K];
    logic signed [DATA_WIDTH-1:0] B [K][N];
    logic signed [ACC_WIDTH-1:0]  C_gold [N][N];

    systolic_array #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) dut (.*);   // ports match TB signal names exactly

    always #5 clk = ~clk;

    //----------------------------------------------------------------
    // Golden model: plain-software matmul. This is the oracle.
    //----------------------------------------------------------------
    function automatic void compute_golden();
        int av, bv;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) begin
                C_gold[i][j] = 0;
                for (int k = 0; k < K; k++) begin
                    av = A[i][k];  if (av > 127) av -= 256;   // manual two's complement
                    bv = B[k][j];  if (bv > 127) bv -= 256;
                    C_gold[i][j] += av * bv;
                end
            end
    endfunction

    //----------------------------------------------------------------
    // Checker: compare every PE's accumulator to the golden result
    //----------------------------------------------------------------
    task automatic check_result(input string name);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (c_out[i][j] !== C_gold[i][j]) begin
                    errors++;
                    $error("[%s] C[%0d][%0d] = %0d, expected %0d",
                           name, i, j, c_out[i][j], C_gold[i][j]);
                end
        if (errors == 0) $display("PASS [%s] all %0d outputs correct", name, N*N);
    endtask

    //----------------------------------------------------------------
    // ============== YOUR PART: the skewed driver ==================
    //
    // Contract: inject A and B as a parallelogram.
    //   - Row i's stream (A[i][0..K-1], valid high) must be delayed
    //     by i cycles relative to row 0.
    //   - Column j's stream (B[0..K-1][j]) delayed by j cycles.
    //   - Before a row's stream starts and after it ends, that row's
    //     valid_in must be LOW (a/b values then don't matter - the
    //     valid gating you verified in T2 protects you).
    //
    // Suggested shape: loop over a global cycle counter t. On each
    // cycle, for each row i decide: "is t within row i's active
    // window [i, i+K-1]? If yes, which element k = t - i do I drive?"
    // Same logic for columns. Drive everything at @(negedge clk)
    // like the PE testbench did.
    //
    // Questions to answer before coding (pencil first!):
    //   1. How many total cycles does injection take? (Last row
    //      starts at t = N-1 and streams K beats.)
    //   2. What is valid_in[i] as a function of t and i?
    //   3. After the last injection, how many cycles until C[N-1][N-1]
    //      is final? (hops to last column + MAC pipeline depth)
    //----------------------------------------------------------------
    task automatic drive_skewed();
        localparam int T = N + K - 1;   // last row starts at N-1, streams K beats

        for (int t = 0; t < T; t++) begin
            @(negedge clk);

            // Rows: A[i][k] where k = t - i, valid only inside the window
            for (int i = 0; i < N; i++) begin
                if (t >= i && t < i + K) begin
                    a_in[i]     = A[i][t-i];
                    valid_in[i] = 1'b1;
                end else begin
                    a_in[i]     = '0;      // your filler zeros
                    valid_in[i] = 1'b0;    // ...but valid low is what matters
                end
            end

            // Columns: B[k][j] where k = t - j
            for (int j = 0; j < N; j++) begin
                if (t >= j && t < j + K)
                    b_in[j] = B[t-j][j];
                else
                    b_in[j] = '0;
            end
        end

        // Final beat: everything idle, or row N-1's last element replays forever
        @(negedge clk);
        for (int i = 0; i < N; i++) begin
            valid_in[i] = 1'b0;
            a_in[i]     = '0;
            b_in[i]     = '0;
        end
    endtask

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    initial begin
        // init & reset
        rst_n = 0; clear = 0; drain = 0;
        for (int i = 0; i < N; i++) begin
            valid_in[i] = 0; a_in[i] = '0; b_in[i] = '0;
        end
        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);
        clear = 1;          // flush the X out of all accumulators
        @(negedge clk);
        clear = 0;
        @(negedge clk);
        //--- Test 1: a fixed, hand-checkable matrix -----------------
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K; k++) begin
                A[i][k] = 8'(i + k + 1);       // simple pattern
                B[k][i] = 8'(1 + ((i * k) % 5));
            end
        compute_golden();

        drive_skewed();

        // TODO: wait the settle time you computed in question 3
        //       (generous overshoot is fine for the first run)
        repeat (20) @(negedge clk);

        check_result("T1 fixed matrix");

        //--- Test 2: clear, then random matrices --------------------
        @(negedge clk) clear = 1;
        @(negedge clk) clear = 0;

        for (int i = 0; i < N; i++)
            for (int k = 0; k < K; k++) begin
                A[i][k] = 8'($urandom);        // full signed range
                B[k][i] = 8'($urandom);
            end
        compute_golden();

        drive_skewed();
        repeat (20) @(negedge clk);
        check_result("T2 random matrix");

        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule