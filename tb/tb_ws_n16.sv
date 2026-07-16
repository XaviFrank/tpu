`timescale 1ns / 1ps
//====================================================================
// tb_ws_n16: WS N=16 scaling checkpoint (verification, not new
// architecture). Instantiates the FULL ws_quant_unit stack at
// N=16 / K=16 (128-bit buffer words) and certifies that everything
// proven at N=4 still holds:
//   T1: 1 vector, identity quant - RAW int32 matvec exact against
//       a 16x16 golden, int8 = clamp, busy = 3N+4+nv = 53 cycles
//       (LOAD N+2=18, STREAM nv+1, FLUSH 2N+1=33), cadence peak 1.
//   T2: 4 vectors back-to-back - 4 in-order results per column
//       (raw + q8), busy = 56, cadence peak = min(n_vecs,N) = 4.
//   T3: 1 vector, relu on - negatives zeroed at the int8 edge.
//   Deterministic formula-generated data (no vector files): weights
//   in [-9,9], activations in [-7,7]; 16-term dot products reach
//   ~±600 so int8 clipping is exercised both ways.
//====================================================================
module tb_ws_n16;

    localparam int N          = 16;
    localparam int K          = 16;
    localparam int DATA_WIDTH = 8;
    localparam int ACC_WIDTH  = 32;
    localparam int OUT_WIDTH  = 8;
    localparam int DEPTH      = 64;   // 4 tiles of 16 words
    localparam int VEC_W      = 4;
    localparam int WORD_W     = N * DATA_WIDTH;   // 128

    // DUT I/O
    logic clk = 1'b0;
    logic rst_n, go;
    logic [$clog2(DEPTH/K)-1:0] w_tile_idx;
    logic [$clog2(DEPTH)-1:0]   a_base;
    logic [VEC_W-1:0]           n_vecs;
    logic                       busy;
    logic                       wr_en_w, wr_en_a;
    logic [$clog2(DEPTH)-1:0]   wr_addr_w, wr_addr_a;
    logic [WORD_W-1:0]          wr_data_w, wr_data_a;
    logic signed [ACC_WIDTH-1:0] sum_out [N];
    logic                        sum_valid [N];
    logic signed [OUT_WIDTH-1:0] q8_out [N];
    logic                        q8_valid [N];
    logic                        relu_en;
    logic signed [15:0]          scale_mult;
    logic [4:0]                  shift_amt;
    logic        perf_clear;
    logic [3:0]  perf_addr;
    logic [31:0] perf_data;

    int errors = 0;

    ws_quant_unit #(
        .N(N), .K(K), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(DEPTH), .VEC_W(VEC_W), .OUT_WIDTH(OUT_WIDTH)
    ) dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .go         (go),
        .w_tile_idx (w_tile_idx),
        .a_base     (a_base),
        .n_vecs     (n_vecs),
        .busy       (busy),
        .perf_clear (perf_clear),
        .perf_addr  (perf_addr),
        .perf_data  (perf_data),
        .wr_en_w    (wr_en_w),
        .wr_addr_w  (wr_addr_w),
        .wr_data_w  (wr_data_w),
        .wr_en_a    (wr_en_a),
        .wr_addr_a  (wr_addr_a),
        .wr_data_a  (wr_data_a),
        .sum_out    (sum_out),
        .sum_valid  (sum_valid),
        .q8_out     (q8_out),
        .q8_valid   (q8_valid),
        .relu_en    (relu_en),
        .scale_mult (scale_mult),
        .shift_amt  (shift_amt)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    // watchdog: a hung unit must fail the regression, not wedge it
    initial begin
        #2_000_000;
        $error("[watchdog] TIMEOUT - unit never finished");
        $finish;
    end

    //----------------------------------------------------------------
    // Deterministic golden data
    //----------------------------------------------------------------
    int w  [16][16];   // resident weights w[i][j]
    int av [4][16];    // av[v][i]

    function automatic int clamp8(input int v, input int relu);
        int r;
        r = v;
        if (relu != 0 && r < 0) r = 0;
        if (r > 127)  r = 127;
        if (r < -128) r = -128;
        return r;
    endfunction

    function automatic int matvec(input int v, input int j);
        int y;
        y = 0;
        for (int i = 0; i < N; i++)
            y = y + w[i][j]*av[v][i];
        return y;
    endfunction

    //----------------------------------------------------------------
    // Loaders: one word = 16 packed int8s, element 0 in [7:0];
    // manual two's complement per byte
    //----------------------------------------------------------------
    task automatic write_w_row(input int tile, input int r);
        logic [WORD_W-1:0] word;
        int b;
        for (int e = 0; e < N; e++) begin
            b = w[r][e];
            if (b < 0) b = b + 256;
            word[e*DATA_WIDTH +: DATA_WIDTH] = b[7:0];
        end
        @(negedge clk) begin
            wr_en_w = 1; wr_addr_w = 6'(tile*K + r); wr_data_w = word;
        end
        @(negedge clk) wr_en_w = 0;
    endtask

    task automatic write_a_vec(input int addr, input int v);
        logic [WORD_W-1:0] word;
        int b;
        for (int e = 0; e < N; e++) begin
            b = av[v][e];
            if (b < 0) b = b + 256;
            word[e*DATA_WIDTH +: DATA_WIDTH] = b[7:0];
        end
        @(negedge clk) begin
            wr_en_a = 1; wr_addr_a = 6'(addr); wr_data_a = word;
        end
        @(negedge clk) wr_en_a = 0;
    endtask

    //----------------------------------------------------------------
    // Collectors (raw + quantized), cadence meter, busy counter
    //----------------------------------------------------------------
    int got8    [16][4];
    int got_raw [16][4];
    int got_cnt [16];
    int cad, cad_max;
    int busy_cycles;

    always @(posedge clk) begin
        if (rst_n) begin
            cad = 0;
            for (int j = 0; j < N; j++) begin
                if (q8_valid[j] === 1'b1) begin
                    got8[j][got_cnt[j]]    = q8_out[j];
                    got_raw[j][got_cnt[j]] = sum_out[j];
                    got_cnt[j] = got_cnt[j] + 1;
                    cad = cad + 1;
                end
            end
            if (cad > cad_max) cad_max = cad;
            if (busy) busy_cycles = busy_cycles + 1;
        end
    end

    //----------------------------------------------------------------
    // Run one tile; check raw + q8 per column/vector, busy budget,
    // and the cadence peak
    //----------------------------------------------------------------
    task automatic run_and_check(input int tile, input int base,
                                 input int nv, input int relu,
                                 input string name);
        int exp_busy, y;
        for (int j = 0; j < N; j++) got_cnt[j] = 0;
        cad_max     = 0;
        busy_cycles = 0;
        exp_busy    = 3*N + 4 + nv;          // LOAD N+2, STREAM nv+1, FLUSH 2N+1

        @(negedge clk) begin
            go = 1; w_tile_idx = tile; a_base = 6'(base);
            n_vecs = VEC_W'(nv); relu_en = relu;
        end
        @(negedge clk) go = 0;
        while (busy) @(negedge clk);
        @(negedge clk);

        for (int j = 0; j < N; j++) begin
            if (got_cnt[j] !== nv) begin
                errors++;
                $error("[%s] column %0d: %0d results, expected %0d", name, j, got_cnt[j], nv);
            end else begin
                for (int v = 0; v < nv; v++) begin
                    y = matvec(v, j);
                    if (got_raw[j][v] !== y) begin
                        errors++;
                        $error("[%s] raw y%0d[%0d] = %0d, expected %0d",
                               name, v, j, got_raw[j][v], y);
                    end
                    if (got8[j][v] !== clamp8(y, relu)) begin
                        errors++;
                        $error("[%s] q8 y%0d[%0d] = %0d, expected %0d (int32 %0d)",
                               name, v, j, got8[j][v], clamp8(y, relu), y);
                    end
                end
            end
        end
        $display("PASS [%s] (%0d columns x %0d vectors, raw+q8)", name, N, nv);

        if (busy_cycles !== exp_busy) begin
            errors++;
            $error("[%s timing] busy = %0d cycles, predicted %0d", name, busy_cycles, exp_busy);
        end else begin
            $display("PASS [%s timing] busy = %0d cycles (3N+4+nv held at N=16)", name, busy_cycles);
        end

        if (cad_max !== ((nv < N) ? nv : N)) begin
            errors++;
            $error("[%s cadence] peak concurrent columns = %0d, predicted %0d",
                   name, cad_max, (nv < N) ? nv : N);
        end else begin
            $display("PASS [%s cadence] peak concurrent columns = %0d", name, cad_max);
        end
    endtask

    //----------------------------------------------------------------
    // Main sequence
    //----------------------------------------------------------------
    initial begin
        rst_n = 0; go = 0; w_tile_idx = '0; a_base = '0; n_vecs = '0;
        wr_en_w = 0; wr_addr_w = '0; wr_data_w = '0;
        wr_en_a = 0; wr_addr_a = '0; wr_data_a = '0;
        relu_en = 0; scale_mult = 16'sd1; shift_amt = '0;   // identity quant
        perf_clear = 0; perf_addr = '0;

        // deterministic fill: signs mixed, magnitudes exercise clipping
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                w[i][j] = ((i*5 + j*3) % 19) - 9;            // -9..9
        for (int v = 0; v < 4; v++)
            for (int i = 0; i < N; i++)
                av[v][i] = ((v*11 + i*7) % 15) - 7;          // -7..7

        repeat (3) @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        // load: weight tile at idx 1 (words 16..31), vectors at 40..43
        for (int r = 0; r < N; r++)
            write_w_row(1, r);
        for (int v = 0; v < 4; v++)
            write_a_vec(40 + v, v);

        //--- T1: single vector, identity quant ------------------------
        run_and_check(1, 40, 1, 0, "T1 N=16 1-vec");

        //--- T2: 4 vectors back-to-back + cadence ---------------------
        run_and_check(1, 40, 4, 0, "T2 N=16 4-vec stream");

        //--- T3: single vector, relu on -------------------------------
        run_and_check(1, 40, 1, 1, "T3 N=16 relu");

        //--- summary ---------------------------------------------------
        repeat (2) @(negedge clk);
        if (errors == 0) $display("\n=== ALL TESTS PASSED ===");
        else             $display("\n=== %0d ERROR(S) ===", errors);
        $finish;
    end

endmodule
