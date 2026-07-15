`timescale 1ns / 1ps
//====================================================================
// tb_matmul_unit v3: black-box test of the BRAM-fed unit
//   Protocol: load tiles through the write ports, pulse go with tile
//   indices, collect labeled rows.
//   T1: fixed tile   T2..T5: random tiles back-to-back
//   T6: go-while-busy ignored + garbage written to a DIFFERENT index
//       mid-flight (verifies index latching too)
//   T7: two tile sets preloaded at different indices, run both
//       back-to-back switching indices only (proves addressing)
//====================================================================
module tb_matmul_unit;

    localparam int N          = 4;
    localparam int K          = 4;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int DEPTH      = 64;
    localparam int NTILES     = DEPTH / K;
    localparam int AW         = $clog2(DEPTH);
    localparam int TW         = $clog2(NTILES);

    logic clk = 0;
    logic rst_n, go;
    logic [TW-1:0]  a_tile_idx, b_tile_idx;
    logic           wr_en_a, wr_en_b;
    logic [AW-1:0]  wr_addr_a, wr_addr_b;
    logic [31:0]    wr_data_a, wr_data_b;
    logic                         busy, result_valid;
    logic [$clog2(N)-1:0]         result_row;
    logic signed [ACC_WIDTH-1:0]  drain_out [N];

    // current tile matrices: source for packing AND for the golden model
    logic signed [DATA_WIDTH-1:0] a_tile [N][K];
    logic signed [DATA_WIDTH-1:0] b_tile [K][N];

    // T7 shelf: first tile set, restored after the second is run
    logic signed [DATA_WIDTH-1:0] a_m0 [N][K];
    logic signed [DATA_WIDTH-1:0] b_m0 [K][N];

    int errors = 0;

    logic signed [ACC_WIDTH-1:0] C_gold [N][N];
    logic signed [ACC_WIDTH-1:0] C_cap  [N][N];

    matmul_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .go           (go),
        .a_tile_idx   (a_tile_idx),
        .b_tile_idx   (b_tile_idx),
        .wr_en_a      (wr_en_a),
        .wr_addr_a    (wr_addr_a),
        .wr_data_a    (wr_data_a),
        .wr_en_b      (wr_en_b),
        .wr_addr_b    (wr_addr_b),
        .wr_data_b    (wr_data_b),
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
    // Load the current a_tile/b_tile matrices into buffer slot idx.
    // Packing per convention: word (idx*K + k) of the a-buffer is
    // column k of A, word (idx*K + k) of the b-buffer is row k of B,
    // element 0 in bits [7:0]. Drive on negedge.
    //----------------------------------------------------------------
    task automatic load_tile_a(input int idx);
        logic [31:0] w;
        for (int k = 0; k < K; k++) begin
            for (int e = 0; e < N; e++)
                w[e*DATA_WIDTH +: DATA_WIDTH] = a_tile[e][k];
            @(negedge clk);
            wr_en_a   = 1'b1;
            wr_addr_a = AW'(idx*K + k);
            wr_data_a = w;
        end
        @(negedge clk) wr_en_a = 1'b0;
    endtask

    task automatic load_tile_b(input int idx);
        logic [31:0] w;
        for (int k = 0; k < K; k++) begin
            for (int e = 0; e < N; e++)
                w[e*DATA_WIDTH +: DATA_WIDTH] = b_tile[k][e];
            @(negedge clk);
            wr_en_b   = 1'b1;
            wr_addr_b = AW'(idx*K + k);
            wr_data_b = w;
        end
        @(negedge clk) wr_en_b = 1'b0;
    endtask

    //----------------------------------------------------------------
    // Run the tiles at (a_idx, b_idx) through the full protocol and
    // check against the golden of the current a_tile/b_tile matrices
    //----------------------------------------------------------------
    task automatic run_tile(input string name, input int a_idx, input int b_idx);
        int errs0;
        compute_golden();

        @(negedge clk);
        go         = 1'b1;
        a_tile_idx = TW'(a_idx);
        b_tile_idx = TW'(b_idx);
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
        a_tile_idx = '0; b_tile_idx = '0;
        wr_en_a = 0; wr_addr_a = '0; wr_data_a = '0;
        wr_en_b = 0; wr_addr_b = '0; wr_data_b = '0;
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
        load_tile_a(0);
        load_tile_b(0);
        run_tile("T1 fixed tile, cold start", 0, 0);

        //--- T2-T5: random tiles, back-to-back, a/b at different slots
        for (int t = 2; t <= 5; t++) begin
            for (int i = 0; i < N; i++)
                for (int k = 0; k < K; k++) begin
                    a_tile[i][k] = 8'($urandom);
                    b_tile[k][i] = 8'($urandom);
                end
            load_tile_a(t);
            load_tile_b(t + 8);
            run_tile($sformatf("T%0d random tile", t), t, t + 8);
        end

        //--- T6: go while busy is ignored ----------------------------
        // Real tiles at a:7/b:9; mid-flight, garbage is written to
        // slot 3 of BOTH buffers and go is hammered with indices
        // pointing at the garbage slot. If go-while-busy were accepted
        // OR the indices weren't latched, the result would corrupt.
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K; k++) begin
                a_tile[i][k] = 8'($urandom);
                b_tile[k][i] = 8'($urandom);
            end
        load_tile_a(7);
        load_tile_b(9);
        compute_golden();
        @(negedge clk);
        go         = 1'b1;
        a_tile_idx = TW'(7);
        b_tile_idx = TW'(9);
        @(negedge clk) go = 1'b0;
        wait (busy);

        // hammer go mid-flight, garbage into a DIFFERENT tile index
        @(negedge clk);
        go         = 1'b1;
        a_tile_idx = TW'(3);
        b_tile_idx = TW'(3);
        for (int k = 0; k < K; k++) begin
            wr_en_a   = 1'b1;
            wr_addr_a = AW'(3*K + k);
            wr_data_a = 32'hx;
            wr_en_b   = 1'b1;
            wr_addr_b = AW'(3*K + k);
            wr_data_b = 32'hx;
            @(negedge clk);
        end
        wr_en_a = 1'b0;
        wr_en_b = 1'b0;
        go      = 1'b0;

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

        //--- T7: two preloaded tile sets, back-to-back by index only -
        // Set 0 -> a:12/b:2, set 1 -> a:4/b:14, loaded up front.
        // Both runs happen with NO reload in between: only the indices
        // change, so this proves addressing, not just the datapath.
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K; k++) begin
                a_tile[i][k] = 8'($urandom);
                b_tile[k][i] = 8'($urandom);
                a_m0[i][k]   = a_tile[i][k];   // shelve set 0
                b_m0[k][i]   = b_tile[k][i];
            end
        load_tile_a(12);
        load_tile_b(2);

        for (int i = 0; i < N; i++)
            for (int k = 0; k < K; k++) begin
                a_tile[i][k] = 8'($urandom);
                b_tile[k][i] = 8'($urandom);
            end
        load_tile_a(4);
        load_tile_b(14);

        run_tile("T7a preloaded set 1 (a:4, b:14)", 4, 14);

        for (int i = 0; i < N; i++)              // restore set 0 for golden
            for (int k = 0; k < K; k++) begin
                a_tile[i][k] = a_m0[i][k];
                b_tile[k][i] = b_m0[k][i];
            end
        run_tile("T7b preloaded set 0 (a:12, b:2), no reload", 12, 2);

        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
