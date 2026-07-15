`timescale 1ns / 1ps
//====================================================================
// tb_matmul_unit v2: black-box test of the self-sequencing unit
//   Protocol only: apply tile, pulse go, collect labeled rows.
//   T1: fixed tile   T2..T5: random tiles back-to-back
//   T6: go-while-busy is ignored
//====================================================================
module tb_matmul_unit;

    localparam int N          = 4;
    localparam int K          = 4;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;

    logic clk = 0;
    logic rst_n, go;
    logic signed [DATA_WIDTH-1:0] a_tile [N][K];
    logic signed [DATA_WIDTH-1:0] b_tile [K][N];
    logic                         busy, result_valid;
    logic [$clog2(N)-1:0]         result_row;
    logic signed [ACC_WIDTH-1:0]  drain_out [N];

    int errors = 0;

    logic signed [ACC_WIDTH-1:0] C_gold [N][N];
    logic signed [ACC_WIDTH-1:0] C_cap  [N][N];

    matmul_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .go           (go),
        .a_tile_in    (a_tile),
        .b_tile_in    (b_tile),
        .busy         (busy),
        .result_valid (result_valid),
        .result_row   (result_row),
        .drain_out    (drain_out)
    );

    always #5 clk = ~clk;

    //----------------------------------------------------------------
    // Passive result collector: whenever the unit says a row is
    // valid, file it under the row index the unit provides.
    // Mid-cycle (negedge) sampling: everything stable.
    //----------------------------------------------------------------
    always @(negedge clk)
        if (result_valid)
            for (int j = 0; j < N; j++)
                C_cap[result_row][j] = drain_out[j];

    //----------------------------------------------------------------
    function automatic void compute_golden();
        int av, bv;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) begin
                C_gold[i][j] = 0;
                for (int k = 0; k < K; k++) begin
                    av = a_tile[i][k];  if (av > 127) av -= 256;   // manual two's complement
                    bv = b_tile[k][j];  if (bv > 127) bv -= 256;
                    C_gold[i][j] += av * bv;
                end
            end
    endfunction

    //----------------------------------------------------------------
    // Run one tile through the full protocol and check it
    //----------------------------------------------------------------
    task automatic run_tile(input string name);
        int errs0;
        compute_golden();

        @(negedge clk) go = 1'b1;
        @(negedge clk) go = 1'b0;

        wait (busy);                 // unit accepted
        wait (!busy);                // unit finished (drain done)
        @(negedge clk);              // let the last capture land

        errs0 = errors;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (C_cap[i][j] !== C_gold[i][j]) begin
                    errors++;
                    $error("[%s] C[%0d][%0d] = %0d, expected %0d",
                           name, i, j, C_cap[i][j], C_gold[i][j]);
                end
        if (errors == errs0) $display("PASS [%s]", name);
    endtask

    //----------------------------------------------------------------
    // Main
    //----------------------------------------------------------------
    initial begin
        rst_n = 0; go = 0;
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K; k++) begin
                a_tile[i][k] = '0;
                b_tile[k][i] = '0;
            end
        repeat (3) @(negedge clk);
        rst_n = 1;
        // NOTE: no clear pulse - S_INIT does it. That's part of the test.

        //--- T1: fixed tile ------------------------------------------
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K; k++) begin
                a_tile[i][k] = 8'(i + k + 1);
                b_tile[k][i] = 8'(1 + ((i * k) % 5));
            end
        run_tile("T1 fixed tile, cold start");

        //--- T2-T5: random tiles, back-to-back -----------------------
        for (int t = 2; t <= 5; t++) begin
            for (int i = 0; i < N; i++)
                for (int k = 0; k < K; k++) begin
                    a_tile[i][k] = 8'($urandom);
                    b_tile[k][i] = 8'($urandom); 
                end
            run_tile($sformatf("T%0d random tile", t));
        end

        //--- T6: go while busy is ignored ----------------------------
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K; k++) begin
                a_tile[i][k] = 8'($urandom);
                b_tile[k][i] = 8'($urandom);
            end
        compute_golden();
        @(negedge clk) go = 1'b1;
        @(negedge clk) go = 1'b0;
        wait (busy);

        // hammer go mid-flight with DIFFERENT (garbage) tile data
        @(negedge clk);
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K; k++) begin
                a_tile[i][k] = 8'shx;
                b_tile[k][i] = 8'shx;
            end
        @(negedge clk) go = 1'b1;
        repeat (3) @(negedge clk);
        go = 1'b0;

        wait (!busy);
        @(negedge clk);
        begin
            int errs0;
            errs0 = errors;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    if (C_cap[i][j] !== C_gold[i][j]) begin
                        errors++;
                        $error("[T6] C[%0d][%0d] = %0d, expected %0d",
                               i, j, C_cap[i][j], C_gold[i][j]);
                    end
            if (errors == errs0) $display("PASS [T6 go-while-busy ignored]");
        end

        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule