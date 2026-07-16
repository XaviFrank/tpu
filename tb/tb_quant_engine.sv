`timescale 1ns / 1ps
//====================================================================
// tb_quant_engine: end-to-end quantized matmul through quant_engine
// (tiling_engine untouched + 4 requant lanes on the read path).
//   Config for the integration checks is the identity quant
//   (scale=1, shift=0), so the golden INT8 is a trivial clamp of
//   the full int matmul golden - the wiring/lanes/packing are what
//   is under test here; the nontrivial scale/round arithmetic is
//   already proven by tb_requant's 64 Python-golden vectors.
//   Q1: 8x8x8 random - raw INT32 row must match golden exactly
//       (wrapper passthrough unperturbed), INT8 row = clamp(C).
//       Then a SECOND readback of the same computed result with
//       relu_en=1: INT8 row = clamp(max(0,C)) - config is
//       combinational at the read port, no recompute needed.
//   Q2: 16x16x16 random (real-size, fills the buffers) with
//       relu_en=0 - the "complete real-size quantized matmul
//       pipeline" artifact of spec 3a.
//====================================================================
module tb_quant_engine;

    localparam int N          = 4;
    localparam int K          = 4;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int OUT_WIDTH  = 8;
    localparam int CHUNK_W    = 4;
    localparam int AB_DEPTH   = 64;
    localparam int OUT_DEPTH  = 64;
    localparam int AW         = $clog2(AB_DEPTH);
    localparam int OW         = $clog2(OUT_DEPTH);
    localparam int MAX_DIM    = 16;

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
    logic [N*OUT_WIDTH-1:0]   out_rd8_data;
    logic                     relu_en;
    logic signed [15:0]       scale_mult;
    logic [4:0]               shift_amt;
    logic                     perf_clear;
    logic [3:0]               perf_addr;
    logic [31:0]              perf_data;

    logic signed [DATA_WIDTH-1:0] A [MAX_DIM][MAX_DIM];
    logic signed [DATA_WIDTH-1:0] B [MAX_DIM][MAX_DIM];
    logic signed [ACC_WIDTH-1:0]  C_gold [MAX_DIM][MAX_DIM];

    int errors = 0;

    quant_engine #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .CHUNK_W(CHUNK_W), .AB_DEPTH(AB_DEPTH), .OUT_DEPTH(OUT_DEPTH),
        .OUT_WIDTH(OUT_WIDTH)
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .m_tiles      (m_tiles),
        .n_tiles      (n_tiles),
        .k_tiles      (k_tiles),
        .done         (done),
        .busy         (busy),
        .wr_en_a      (wr_en_a),
        .wr_addr_a    (wr_addr_a),
        .wr_data_a    (wr_data_a),
        .wr_en_b      (wr_en_b),
        .wr_addr_b    (wr_addr_b),
        .wr_data_b    (wr_data_b),
        .out_rd_en    (out_rd_en),
        .out_rd_addr  (out_rd_addr),
        .out_rd_data  (out_rd_data),
        .out_rd8_data (out_rd8_data),
        .relu_en      (relu_en),
        .scale_mult   (scale_mult),
        .shift_amt    (shift_amt),
        .perf_clear   (perf_clear),
        .perf_addr    (perf_addr),
        .perf_data    (perf_data)
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

    // identity-quant golden: clamp (+ optional ReLU) of the int32 value
    function automatic int clamp8(input int v, input int relu);
        int r;
        r = v;
        if (relu != 0 && r < 0) r = 0;
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        return r;
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
    // Readback of the whole C region under the CURRENT quant config;
    // checks the int8 lanes, and optionally the raw int32 row too
    //----------------------------------------------------------------
    task automatic check_readback(input string name, input int mt,
                                  input int nt, input int relu,
                                  input int check_raw);
        int errs0, gi, gj, got8;
        logic [N*ACC_WIDTH-1:0] w;
        logic [N*OUT_WIDTH-1:0] w8;

        errs0 = errors;
        for (int ti = 0; ti < mt; ti++)
            for (int tj = 0; tj < nt; tj++)
                for (int r = 0; r < N; r++) begin
                    @(negedge clk);
                    out_rd_en   = 1'b1;
                    out_rd_addr = OW'((ti*nt + tj)*N + r);
                    @(negedge clk);
                    out_rd_en = 1'b0;
                    w  = out_rd_data;
                    w8 = out_rd8_data;
                    for (int lane = 0; lane < N; lane++) begin
                        gi = ti*N + r;
                        gj = tj*N + lane;
                        if (check_raw != 0 &&
                            w[lane*ACC_WIDTH +: ACC_WIDTH] !== C_gold[gi][gj]) begin
                            errors++;
                            $error("[%s] raw C[%0d][%0d] = %0d, expected %0d",
                                   name, gi, gj,
                                   $signed(w[lane*ACC_WIDTH +: ACC_WIDTH]),
                                   C_gold[gi][gj]);
                        end
                        got8 = w8[lane*OUT_WIDTH +: OUT_WIDTH];
                        if (got8 > 127) got8 -= 256;   // manual two's complement
                        if (got8 !== clamp8(C_gold[gi][gj], relu)) begin
                            errors++;
                            $error("[%s] q8 C[%0d][%0d] = %0d, expected %0d (int32 %0d)",
                                   name, gi, gj, got8,
                                   clamp8(C_gold[gi][gj], relu), C_gold[gi][gj]);
                        end
                    end
                end
        if (errors == errs0) $display("PASS [%s]", name);
    endtask

    task automatic run_engine(input int mt, input int kt, input int nt);
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
        wait (done);
        wait (!busy);
        @(negedge clk);
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
        relu_en = 0; scale_mult = 16'sd1; shift_amt = '0;   // identity quant
        for (int i = 0; i < MAX_DIM; i++)
            for (int j = 0; j < MAX_DIM; j++) begin
                A[i][j] = '0;
                B[i][j] = '0;
            end
        repeat (3) @(negedge clk);
        rst_n = 1;

        //--- Q1: 8x8x8 random; both ReLU modes off one compute --------
        fill_random();
        run_engine(2, 2, 2);
        relu_en = 0;
        check_readback("Q1a 8x8x8 raw+q8, relu off", 2, 2, 0, 1);
        relu_en = 1;                       // config change only, no recompute
        check_readback("Q1b same result, relu on", 2, 2, 1, 0);

        //--- Q2: 16x16x16 random real-size (spec 3a artifact) ---------
        relu_en = 0;
        fill_random();
        run_engine(4, 4, 4);
        check_readback("Q2 16x16x16 quantized", 4, 4, 0, 1);

        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
