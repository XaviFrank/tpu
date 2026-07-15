`timescale 1ns / 1ps
//====================================================================
// tb_tiling_engine: end-to-end tile-multiple matmuls
//   Load full A and B through the pass-through write ports, pulse
//   start with sizes in tiles, wait for done, read the whole output
//   buffer back through its read port (1-cycle latency), compare
//   every element against a full MxKxN int golden model.
//   E1: 8x8x8 fixed hand-checkable (B = identity, so C = A)
//   E2: 8x8x8 random     E3: 16x16x16 random (fills the buffers)
//   E4: 8x16x4 random (non-cubic: unequal counters + address math)
//   E5: two consecutive starts without reset, fresh data each
//   E3 also dumps the perf counters (utilization baseline for
//   docs/perf_baseline.md; AUTO_CLEAR makes them per-run).
//====================================================================
module tb_tiling_engine;

    localparam int N          = 4;
    localparam int K          = 4;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int CHUNK_W    = 4;
    localparam int AB_DEPTH   = 64;
    localparam int OUT_DEPTH  = 64;
    localparam int AW         = $clog2(AB_DEPTH);
    localparam int OW         = $clog2(OUT_DEPTH);
    localparam int MAX_DIM    = 16;   // buffers hold up to 4x4 tiles per matrix

    logic clk = 0;
    logic rst_n, start;
    logic [3:0] m_tiles, n_tiles, k_tiles;
    logic done, busy;
    logic          wr_en_a, wr_en_b;
    logic [AW-1:0] wr_addr_a, wr_addr_b;
    logic [31:0]   wr_data_a, wr_data_b;
    logic                     out_rd_en;
    logic [OW-1:0]            out_rd_addr;
    logic [N*ACC_WIDTH-1:0]   out_rd_data;
    logic                     perf_clear;
    logic [3:0]               perf_addr;
    logic [31:0]              perf_data;

    // full matrices (used region set per test)
    logic signed [DATA_WIDTH-1:0] A [MAX_DIM][MAX_DIM];
    logic signed [DATA_WIDTH-1:0] B [MAX_DIM][MAX_DIM];
    logic signed [ACC_WIDTH-1:0]  C_gold [MAX_DIM][MAX_DIM];

    int errors = 0;

    tiling_engine #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .CHUNK_W(CHUNK_W), .AB_DEPTH(AB_DEPTH), .OUT_DEPTH(OUT_DEPTH)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .start       (start),
        .m_tiles     (m_tiles),
        .n_tiles     (n_tiles),
        .k_tiles     (k_tiles),
        .done        (done),
        .busy        (busy),
        .wr_en_a     (wr_en_a),
        .wr_addr_a   (wr_addr_a),
        .wr_data_a   (wr_data_a),
        .wr_en_b     (wr_en_b),
        .wr_addr_b   (wr_addr_b),
        .wr_data_b   (wr_data_b),
        .out_rd_en   (out_rd_en),
        .out_rd_addr (out_rd_addr),
        .out_rd_data (out_rd_data),
        .perf_clear  (perf_clear),
        .perf_addr   (perf_addr),
        .perf_data   (perf_data)
    );

    always #5 clk = ~clk;

    // watchdog: a hung engine must fail the regression, not wedge it
    initial begin
        #5_000_000;
        $error("[watchdog] TIMEOUT - engine never finished");
        $finish;
    end

    //----------------------------------------------------------------
    // Golden: full MxKxN int matmul, manual two's complement
    //----------------------------------------------------------------
    function automatic void compute_golden(input int M, input int Kd, input int Nd);
        int av, bv;
        for (int i = 0; i < M; i++)
            for (int j = 0; j < Nd; j++) begin
                C_gold[i][j] = 0;
                for (int k = 0; k < Kd; k++) begin
                    av = A[i][k];  if (av > 127) av -= 256;   // manual two's complement
                    bv = B[k][j];  if (bv > 127) bv -= 256;
                    C_gold[i][j] += av * bv;
                end
            end
    endfunction

    task automatic fill_random();
        for (int i = 0; i < MAX_DIM; i++)
            for (int j = 0; j < MAX_DIM; j++) begin
                A[i][j] = 8'($urandom);
                B[i][j] = 8'($urandom);
            end
    endtask

    //----------------------------------------------------------------
    // Loaders: pack per the beat convention (element 0 in bits [7:0])
    //   A tile (I,G) at a_tile_idx = I*kt + G, word k = column k
    //   B tile (G,J) at b_tile_idx = G*nt + J, word k = row k
    //----------------------------------------------------------------
    task automatic load_A(input int mt, input int kt);
        logic [31:0] w;
        for (int ti = 0; ti < mt; ti++)
            for (int tg = 0; tg < kt; tg++)
                for (int k = 0; k < K; k++) begin
                    for (int e = 0; e < N; e++)
                        w[e*DATA_WIDTH +: DATA_WIDTH] = A[ti*N + e][tg*K + k];
                    @(negedge clk);
                    wr_en_a   = 1'b1;
                    wr_addr_a = AW'((ti*kt + tg)*K + k);
                    wr_data_a = w;
                end
        @(negedge clk) wr_en_a = 1'b0;
    endtask

    task automatic load_B(input int kt, input int nt);
        logic [31:0] w;
        for (int tg = 0; tg < kt; tg++)
            for (int tj = 0; tj < nt; tj++)
                for (int k = 0; k < K; k++) begin
                    for (int e = 0; e < N; e++)
                        w[e*DATA_WIDTH +: DATA_WIDTH] = B[tg*K + k][tj*N + e];
                    @(negedge clk);
                    wr_en_b   = 1'b1;
                    wr_addr_b = AW'((tg*nt + tj)*K + k);
                    wr_data_b = w;
                end
        @(negedge clk) wr_en_b = 1'b0;
    endtask

    //----------------------------------------------------------------
    // Perf access. read_perf CONTRACT: call at a negedge; address
    // presented mid-cycle, DUT registers at the next posedge, value
    // sampled at the next negedge (1 cycle per call).
    //----------------------------------------------------------------
    task automatic read_perf(input int addr, output logic [31:0] val);
        perf_addr = 4'(addr);
        @(negedge clk);
        val = perf_data;
    endtask

    task automatic dump_perf(input string tag);
        logic [31:0] v0, v1, v2, v3, v4, v5, v6, v7;
        int util_x10;
        read_perf(0, v0);
        read_perf(1, v1);
        read_perf(2, v2);
        read_perf(3, v3);
        read_perf(4, v4);
        read_perf(5, v5);
        read_perf(6, v6);
        read_perf(7, v7);
        $display("[PERF %s] total=%0d fetch=%0d feed=%0d settle=%0d drain=%0d idle=%0d chunks=%0d tiles=%0d",
                 tag, v0, v1, v2, v3, v4, v5, v6, v7);
        util_x10 = (v2 * 1000) / v0;
        $display("[PERF %s] utilization = feed/total = %0d.%0d%%",
                 tag, util_x10 / 10, util_x10 % 10);
    endtask

    //----------------------------------------------------------------
    // Full round trip: load, start, wait done, (optional perf dump
    // right after completion, before readback pollutes idle/total),
    // read back, compare
    //----------------------------------------------------------------
    task automatic run_case(input string name, input int mt, input int kt,
                            input int nt, input int do_dump);
        int errs0;
        int gi, gj;
        logic [N*ACC_WIDTH-1:0] w;

        load_A(mt, kt);
        load_B(kt, nt);
        compute_golden(mt*N, kt*K, nt*N);

        @(negedge clk);
        start   = 1'b1;
        m_tiles = 4'(mt);
        n_tiles = 4'(nt);
        k_tiles = 4'(kt);
        @(negedge clk) start = 1'b0;

        wait (busy);
        wait (done);                 // done must pulse (1 cycle)
        wait (!busy);
        @(negedge clk);

        if (do_dump != 0)
            dump_perf(name);

        errs0 = errors;
        for (int ti = 0; ti < mt; ti++)
            for (int tj = 0; tj < nt; tj++)
                for (int r = 0; r < N; r++) begin
                    @(negedge clk);
                    out_rd_en   = 1'b1;
                    out_rd_addr = OW'((ti*nt + tj)*N + r);
                    @(negedge clk);
                    out_rd_en = 1'b0;
                    w = out_rd_data;
                    for (int lane = 0; lane < N; lane++) begin
                        gi = ti*N + r;
                        gj = tj*N + lane;
                        if (w[lane*ACC_WIDTH +: ACC_WIDTH] !== C_gold[gi][gj]) begin
                            errors++;
                            $error("[%s] C[%0d][%0d] = %0d, expected %0d",
                                   name, gi, gj,
                                   $signed(w[lane*ACC_WIDTH +: ACC_WIDTH]),
                                   C_gold[gi][gj]);
                        end
                    end
                end
        if (errors == errs0) $display("PASS [%s]", name);
    endtask

    //----------------------------------------------------------------
    // Main
    //----------------------------------------------------------------
    initial begin
        rst_n = 0; start = 0;
        m_tiles = '0; n_tiles = '0; k_tiles = '0;
        wr_en_a = 0; wr_addr_a = '0; wr_data_a = '0;
        wr_en_b = 0; wr_addr_b = '0; wr_data_b = '0;
        out_rd_en = 0; out_rd_addr = '0;
        perf_clear = 0; perf_addr = '0;
        for (int i = 0; i < MAX_DIM; i++)
            for (int j = 0; j < MAX_DIM; j++) begin
                A[i][j] = '0;
                B[i][j] = '0;
            end
        repeat (3) @(negedge clk);
        rst_n = 1;

        //--- E1: 8x8x8, fixed hand-checkable: B = I, so C = A --------
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++) begin
                A[i][j] = 8'(i - j);              // negatives included
                B[i][j] = (i == j) ? 8'sd1 : 8'sd0;
            end
        run_case("E1 8x8x8 fixed, B=identity", 2, 2, 2, 0);

        //--- E2: 8x8x8 random ----------------------------------------
        fill_random();
        run_case("E2 8x8x8 random", 2, 2, 2, 0);

        //--- E3: 16x16x16 random (fills all three buffers) ------------
        //    perf dump = the utilization measurement of record
        fill_random();
        run_case("E3 16x16x16 random", 4, 4, 4, 1);

        //--- E4: 8x16x4 random (non-cubic, m=2 k=4 n=1 tiles) ---------
        fill_random();
        run_case("E4 8x16x4 random", 2, 4, 1, 0);

        //--- E5: two consecutive starts, no reset, fresh data ---------
        fill_random();
        run_case("E5a consecutive start #1", 2, 2, 2, 0);
        fill_random();
        run_case("E5b consecutive start #2", 2, 2, 2, 0);

        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
