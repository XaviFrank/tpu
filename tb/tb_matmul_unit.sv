`timescale 1ns / 1ps
//====================================================================
// tb_matmul_unit v4: black-box test of the BRAM-fed K-walk unit
//   Protocol: load tiles through the write ports, pulse go with
//   k_chunks + first tile indices, answer chunk_req with the next
//   chunk's indices, collect labeled rows.
//   T1: fixed tile   T2..T5: random tiles back-to-back (k_chunks=1)
//   T6: go-while-busy ignored + garbage written to a DIFFERENT index
//       mid-flight (verifies index AND k_chunks latching)
//   T7: two tile sets preloaded at different indices, run both
//       back-to-back switching indices only (proves addressing)
//   T8: k_chunks=2 walk vs golden at K=8
//   T9: k_chunks=4 walk vs golden at K=16, plus an explicit check
//       that the result is NOT just the final chunk's contribution
//   T10: k_chunks=1 back-to-back with k_chunks=3 - length is per-go
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
    localparam int CHUNK_W    = 4;

    logic clk = 0;
    logic rst_n, go;
    logic [CHUNK_W-1:0] k_chunks;
    logic               chunk_req;
    logic [TW-1:0]  a_tile_idx, b_tile_idx;
    logic           wr_en_a, wr_en_b;
    logic [AW-1:0]  wr_addr_a, wr_addr_b;
    logic [31:0]    wr_data_a, wr_data_b;
    logic                         busy, result_valid;
    logic [$clog2(N)-1:0]         result_row;
    logic signed [ACC_WIDTH-1:0]  drain_out [N];

    // perf flags: unused here (observed by tb_tiling_engine's counters)
    logic mm_in_fetch, mm_in_feed, mm_in_settle, mm_in_drain, mm_in_idle;
    logic mm_chunk_done, mm_tile_done;

    // current chunk's tile matrices: source for packing AND golden
    logic signed [DATA_WIDTH-1:0] a_tile [N][K];
    logic signed [DATA_WIDTH-1:0] b_tile [K][N];

    // T7 shelf: first tile set, restored after the second is run
    logic signed [DATA_WIDTH-1:0] a_m0 [N][K];
    logic signed [DATA_WIDTH-1:0] b_m0 [K][N];

    int errors = 0;

    logic signed [ACC_WIDTH-1:0] C_gold [N][N];
    logic signed [ACC_WIDTH-1:0] C_last [N][N];   // final chunk alone (T9)
    logic signed [ACC_WIDTH-1:0] C_cap  [N][N];

    matmul_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .CHUNK_W(CHUNK_W)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .go           (go),
        .k_chunks     (k_chunks),
        .chunk_req    (chunk_req),
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
        .drain_out    (drain_out),
        .in_fetch     (mm_in_fetch),
        .in_feed      (mm_in_feed),
        .in_settle    (mm_in_settle),
        .in_drain     (mm_in_drain),
        .in_idle      (mm_in_idle),
        .chunk_done   (mm_chunk_done),
        .tile_done    (mm_tile_done)
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
    // Golden model: C_gold accumulates a_tile*b_tile per chunk
    // (int locals, manual two's complement)
    //----------------------------------------------------------------
    function automatic void clear_golden();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                C_gold[i][j] = 0;
    endfunction

    function automatic void accumulate_golden();
        int av, bv;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < K; k++) begin
                    av = a_tile[i][k];  if (av > 127) av -= 256;   // manual two's complement
                    bv = b_tile[k][j];  if (bv > 127) bv -= 256;
                    C_gold[i][j] += av * bv;
                end
    endfunction

    function automatic void compute_golden();
        clear_golden();
        accumulate_golden();
    endfunction

    // C_last = product of the CURRENT a_tile/b_tile alone (T9 check)
    function automatic void compute_last();
        int av, bv;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) begin
                C_last[i][j] = 0;
                for (int k = 0; k < K; k++) begin
                    av = a_tile[i][k];  if (av > 127) av -= 256;
                    bv = b_tile[k][j];  if (bv > 127) bv -= 256;
                    C_last[i][j] += av * bv;
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

    // fill the current chunk matrices with fresh random data
    task automatic randomize_tiles();
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K; k++) begin
                a_tile[i][k] = 8'($urandom);
                b_tile[k][i] = 8'($urandom);
            end
    endtask

    //----------------------------------------------------------------
    // Run one K-walk: chunk g's tiles must already be loaded at
    // a_base+g / b_base+g, and C_gold already computed. Presents
    // chunk 0's indices with go, then answers each chunk_req window
    // (sampled mid-cycle on negedge; controller registers at the
    // posedge that ends the window). Checks C_cap vs C_gold.
    //----------------------------------------------------------------
    task automatic run_walk(input string name, input int nchunks,
                            input int a_base, input int b_base);
        int errs0;

        @(negedge clk);
        go         = 1'b1;
        k_chunks   = CHUNK_W'(nchunks);
        a_tile_idx = TW'(a_base);
        b_tile_idx = TW'(b_base);
        @(negedge clk) go = 1'b0;

        wait (busy);                 // unit accepted
        for (int g = 1; g < nchunks; g++) begin
            while (chunk_req !== 1'b1) @(negedge clk);
            a_tile_idx = TW'(a_base + g);
            b_tile_idx = TW'(b_base + g);
            @(negedge clk);          // move past this window
        end
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

    // single-tile convenience: golden from current matrices, 1 chunk
    task automatic run_tile(input string name, input int a_idx, input int b_idx);
        compute_golden();
        run_walk(name, 1, a_idx, b_idx);
    endtask

    //----------------------------------------------------------------
    // Main
    //----------------------------------------------------------------
    initial begin
        rst_n = 0; go = 0;
        k_chunks   = CHUNK_W'(1);
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
            randomize_tiles();
            load_tile_a(t);
            load_tile_b(t + 8);
            run_tile($sformatf("T%0d random tile", t), t, t + 8);
        end

        //--- T6: go while busy is ignored ----------------------------
        // Real tiles at a:7/b:9; mid-flight, garbage is written to
        // slot 3 of BOTH buffers and go is hammered with indices
        // pointing at the garbage slot AND a bogus k_chunks. If
        // go-while-busy were accepted, or the indices / walk length
        // weren't latched, the result would corrupt.
        randomize_tiles();
        load_tile_a(7);
        load_tile_b(9);
        compute_golden();
        @(negedge clk);
        go         = 1'b1;
        k_chunks   = CHUNK_W'(1);
        a_tile_idx = TW'(7);
        b_tile_idx = TW'(9);
        @(negedge clk) go = 1'b0;
        wait (busy);

        // hammer go mid-flight, garbage into a DIFFERENT tile index
        @(negedge clk);
        go         = 1'b1;
        k_chunks   = 4'hF;
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
        wr_en_a  = 1'b0;
        wr_en_b  = 1'b0;
        go       = 1'b0;
        k_chunks = CHUNK_W'(1);

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
        randomize_tiles();
        for (int i = 0; i < N; i++)
            for (int k = 0; k < K; k++) begin
                a_m0[i][k] = a_tile[i][k];     // shelve set 0
                b_m0[k][i] = b_tile[k][i];
            end
        load_tile_a(12);
        load_tile_b(2);

        randomize_tiles();
        load_tile_a(4);
        load_tile_b(14);

        run_tile("T7a preloaded set 1 (a:4, b:14)", 4, 14);

        for (int i = 0; i < N; i++)            // restore set 0 for golden
            for (int k = 0; k < K; k++) begin
                a_tile[i][k] = a_m0[i][k];
                b_tile[k][i] = b_m0[k][i];
            end
        run_tile("T7b preloaded set 0 (a:12, b:2), no reload", 12, 2);

        //--- T8: k_chunks=2 = one logical K=8 matmul ------------------
        // A (4x8) split into chunks a:1,a:2; B (8x4) into b:5,b:6.
        // Golden accumulates both chunks - exactly K=8 semantics.
        clear_golden();
        for (int g = 0; g < 2; g++) begin
            randomize_tiles();
            load_tile_a(1 + g);
            load_tile_b(5 + g);
            accumulate_golden();
        end
        run_walk("T8 k_chunks=2 (K=8 walk)", 2, 1, 5);

        //--- T9: k_chunks=4 = one logical K=16 matmul (the killer) ----
        clear_golden();
        for (int g = 0; g < 4; g++) begin
            randomize_tiles();
            load_tile_a(10 + g);
            load_tile_b(1 + g);
            accumulate_golden();
            if (g == 3) compute_last();        // final chunk alone
        end
        run_walk("T9 k_chunks=4 (K=16 walk)", 4, 10, 1);

        // If the FEED->FETCH loop-back accidentally cleared the
        // accumulators, C_cap would equal ONLY the final chunk's
        // contribution. Assert that is not the case.
        begin
            int same;
            same = 1;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    if (C_cap[i][j] !== C_last[i][j]) same = 0;
            if (same) begin
                errors++;
                $error("[T9] result equals the FINAL CHUNK alone - accumulators were cleared across chunks");
            end else begin
                $display("PASS [T9b partials carried across chunks]");
            end
        end

        //--- T10: walk length is per-go: 1, then 3, then 1 ------------
        randomize_tiles();
        load_tile_a(0);
        load_tile_b(0);
        run_tile("T10a k_chunks=1", 0, 0);

        clear_golden();
        for (int g = 0; g < 3; g++) begin
            randomize_tiles();
            load_tile_a(4 + g);
            load_tile_b(8 + g);
            accumulate_golden();
        end
        run_walk("T10b k_chunks=3", 3, 4, 8);

        randomize_tiles();
        load_tile_a(15);
        load_tile_b(15);
        run_tile("T10c k_chunks=1 again", 15, 15);

        //--- T11: chunk_req contract: combinational answer -------------
        // CONTRACT NOTE: the handshake is COMBINATIONAL-ANSWER - the
        // issuer presents the next chunk's indices in the same cycle
        // chunk_req is high (this TB answers at the mid-cycle negedge;
        // tiling_engine answers combinationally from its counters,
        // i.e. a 0-cycle delay). If a delayed-answer mode is ever
        // needed, that is a spec change, not a TB tweak.
        clear_golden();
        for (int g = 0; g < 3; g++) begin
            randomize_tiles();
            load_tile_a(5 + g);
            load_tile_b(12 + g);
            accumulate_golden();
        end
        run_walk("T11 k_chunks=3, combinational chunk_req answer", 3, 5, 12);

        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
